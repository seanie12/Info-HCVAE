import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from model.customized_layers import ContextualizedEmbedding, Embedding
from model.encoders import PosteriorEncoder, PriorEncoder
from model.decoders import QuestionDecoder, AnswerDecoder
from model.losses import GaussianKLLoss, CategoricalKLLoss, ContinuousKernelMMDLoss, CategoricalMMDLoss
from model.infomax import AnswerSpanInfoMaxLoss


class DiscreteVAE(nn.Module):
    def __init__(self, args):
        super(DiscreteVAE, self).__init__()
        tokenizer = BertTokenizer.from_pretrained(args.huggingface_model)
        padding_idx = tokenizer.vocab['[PAD]']
        sos_id = tokenizer.vocab['[CLS]']
        eos_id = tokenizer.vocab['[SEP]']
        ntokens = len(tokenizer.vocab)

        huggingface_model = args.huggingface_model
        if "large" in huggingface_model:
            emsize = 1024
        else:
            emsize = 768

        enc_nhidden = args.enc_nhidden
        enc_nlayers = args.enc_nlayers
        enc_dropout = args.enc_dropout
        dec_a_nhidden = args.dec_a_nhidden
        dec_a_nlayers = args.dec_a_nlayers
        dec_a_dropout = args.dec_a_dropout
        self.dec_q_nhidden = dec_q_nhidden = args.dec_q_nhidden
        self.dec_q_nlayers = dec_q_nlayers = args.dec_q_nlayers
        dec_q_dropout = args.dec_q_dropout
        self.nzqdim = nzqdim = args.nzqdim
        self.nza = nza = args.nza
        self.nzadim = nzadim = args.nzadim

        self.w_bce = args.w_bce
        self.alpha_kl = args.alpha_kl
        self.gamma_span_info = args.gamma_span_info
        self.lambda_qa_info = args.lambda_qa_info

        max_q_len = args.max_q_len

        embedding = Embedding(huggingface_model)
        contextualized_embedding = ContextualizedEmbedding(huggingface_model)
        # freeze embedding
        for param in embedding.parameters():
            param.requires_grad = False
        for param in contextualized_embedding.parameters():
            param.requires_grad = False

        self.posterior_encoder = PosteriorEncoder(embedding, emsize,
                                                  enc_nhidden, enc_nlayers,
                                                  nzqdim, nza, nzadim,
                                                  enc_dropout)

        self.prior_encoder = PriorEncoder(embedding, emsize,
                                          enc_nhidden, enc_nlayers,
                                          nzqdim, nza, nzadim, enc_dropout)

        self.answer_decoder = AnswerDecoder(contextualized_embedding, emsize,
                                            dec_a_nhidden, dec_a_nlayers,
                                            dec_a_dropout)

        self.question_decoder = QuestionDecoder(sos_id, eos_id,
                                                embedding, contextualized_embedding, emsize,
                                                dec_q_nhidden, ntokens, dec_q_nlayers,
                                                dec_q_dropout, max_q_len)

        self.q_h_linear = nn.Linear(nzqdim, dec_q_nlayers * dec_q_nhidden)
        self.q_c_linear = nn.Linear(nzqdim, dec_q_nlayers * dec_q_nhidden)
        self.a_linear = nn.Linear(nza*nzadim, emsize*args.max_c_len, False)

        self.q_rec_criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
        self.a_rec_criterion = nn.CrossEntropyLoss(ignore_index=args.max_c_len)
        self.gaussian_kl_criterion = GaussianKLLoss()
        self.categorical_kl_criterion = CategoricalKLLoss()

        if self.gamma_span_info > 0:
            # enc_nhidden * 2 to account for bidirectional case
            # self.ans_global_infomax = AnswerMutualInfoMax(
            #     2*dec_a_nhidden, 2*dec_a_nhidden, infomax_type="bce")
            # self.ans_local_infomax = AnswerMutualInfoMax(
            #     2*dec_a_nhidden, 2*dec_a_nhidden, infomax_type="bce")
            # self.ans_global_infomax.denote_is_infomax_net_for_params()
            self.ans_global_infomax = AnswerSpanInfoMaxLoss(2*dec_a_nhidden)
            self.ans_local_infomax = AnswerSpanInfoMaxLoss(2*dec_a_nhidden)

    def return_init_state(self, zq, za):
        q_init_h = F.mish(self.q_h_linear(zq))
        q_init_c = F.mish(self.q_c_linear(zq))
        q_init_h = q_init_h.view(-1, self.dec_q_nlayers,
                                 self.dec_q_nhidden).transpose(0, 1).contiguous()
        q_init_c = q_init_c.view(-1, self.dec_q_nlayers,
                                 self.dec_q_nhidden).transpose(0, 1).contiguous()
        q_init_state = (q_init_h, q_init_c)

        za_flatten = za.view(-1, self.nza * self.nzadim)
        a_init_state = F.mish(self.a_linear(za_flatten))

        return q_init_state, a_init_state

    def forward(self, c_ids, q_ids, a_ids, start_positions, end_positions):
        posterior_zq_mu, posterior_zq_logvar, posterior_zq, \
            posterior_za_logits, posterior_za \
            = self.posterior_encoder(c_ids, q_ids, a_ids)

        prior_zq_mu, prior_zq_logvar, _, \
            prior_za_logits, _ \
            = self.prior_encoder(c_ids)

        q_init_state, a_init_state = self.return_init_state(
            posterior_zq, posterior_za)

        # answer decoding
        start_logits, end_logits, dec_ans_outputs = self.answer_decoder(
            a_init_state, c_ids)
        # question decoding
        q_logits, loss_info = self.question_decoder(
            q_init_state, c_ids, q_ids, a_ids)

        # Compute losses
        if self.training:
            # q rec loss
            loss_q_rec = self.q_rec_criterion(q_logits[:, :-1, :].transpose(1, 2).contiguous(),
                                              q_ids[:, 1:])

            # a rec loss
            max_c_len = c_ids.size(1)
            start_positions.clamp_(0, max_c_len)
            end_positions.clamp_(0, max_c_len)
            loss_start_a_rec = self.a_rec_criterion(
                start_logits, start_positions)
            loss_end_a_rec = self.a_rec_criterion(end_logits, end_positions)
            loss_a_rec = 0.5 * (loss_start_a_rec + loss_end_a_rec)

            # kl loss
            loss_zq_kl = self.gaussian_kl_criterion(posterior_zq_mu, posterior_zq_logvar,
                                                    prior_zq_mu, prior_zq_logvar)

            loss_za_kl = self.categorical_kl_criterion(posterior_za_logits,
                                                       prior_za_logits)

            loss_span_info = torch.tensor(0, device=loss_zq_kl.device)
            if self.gamma_span_info > 0:
                ### compute QAInfomax-based regularizer loss ###
                # dec_ans_outputs.size() = (N, seq_len, 2*dec_a_nhidden)

                # context means set of {paragraph embeddings}. answer decoding does not require question embeddings
                context_enc = []
                ans_enc = []
                sampled_word_range = [] # for LC computation
                batch_size = dec_ans_outputs.size(0)
                for b_idx in range(batch_size):
                    # invalid example or impossible example
                    if start_positions[b_idx].item() == 0 and end_positions[b_idx].item() == 0:
                        continue

                    context_output = torch.cat((dec_ans_outputs[b_idx, :start_positions[b_idx], :],
                                                dec_ans_outputs[b_idx, end_positions[b_idx] + 1:, :]),
                                               dim=0)
                    context_enc.append(context_output.unsqueeze(0))

                    # extend by 5 tokens to take into account local context
                    extend_start = max(0, start_positions[b_idx] - 5)
                    extend_end = min(dec_ans_outputs.size(1),
                                     end_positions[b_idx] + 5)
                    # sample a word from answer span to compute LC
                    start_idx = abs(start_positions[b_idx] - extend_start)
                    end_idx = extend_end - extend_start - abs(extend_end - end_positions[b_idx])
                    sampled_word_range.append((start_idx, end_idx))
                    # (seq, hidden_size)
                    ans_seq = dec_ans_outputs[b_idx,
                                              extend_start: extend_end, :]
                    ans_enc.append(ans_seq.unsqueeze(0))
                assert len(ans_enc) == len(context_enc)

                # generate fake examples by shifting
                shift = random.randint(1, len(context_enc))
                context_fake = context_enc[-shift:] + context_enc[:-shift]
                ans_fake = ans_enc[-shift:] + ans_enc[:-shift]

                global_loss = 0
                local_loss = 0
                for b_idx in range(len(context_enc)):
                    c_enc, c_fake = context_enc[b_idx], context_fake[b_idx]
                    a_enc, a_fake = ans_enc[b_idx], ans_fake[b_idx]

                    ## Compute GC ##
                    global_loss = global_loss + self.ans_global_infomax(a_enc, a_fake,
                                                                        c_enc, c_fake, do_summarize=True)

                    ## Compute LC ##
                    # sample one
                    rand_idx = random.randint(sampled_word_range[b_idx][0], sampled_word_range[b_idx][1] - 1)
                    a_enc_word = a_enc[0, rand_idx, :]
                    rand_idx = random.randint(0, a_fake.size(1) - 1)
                    a_enc_fake = a_fake[0, rand_idx, :]
                    local_loss = local_loss + self.ans_local_infomax(a_enc_word.unsqueeze(0),
                                                                     a_enc_fake.unsqueeze(0), a_enc, a_fake, do_summarize=False)

                loss_span_info = self.gamma_span_info * \
                    (0.5 * global_loss + local_loss) / len(ans_enc)

            loss_kl = self.alpha_kl * (loss_zq_kl + loss_za_kl)
            loss_qa_info = self.lambda_qa_info * loss_info
            loss = self.w_bce * (loss_q_rec + loss_a_rec) + \
                loss_kl + loss_qa_info + loss_span_info

            return_dict = {
                "total_loss": loss,
                "loss_q_rec": loss_q_rec,
                "loss_a_rec": loss_a_rec,
                "loss_kl": loss_kl,
                "loss_zq_kl": loss_zq_kl,
                "loss_za_kl": loss_za_kl,
                "loss_span_info": loss_span_info,
                "loss_qa_info": loss_qa_info,
            }
            return return_dict

    def generate(self, zq, za, c_ids):
        q_init_state, a_init_state = self.return_init_state(zq, za)

        a_ids, start_positions, end_positions = self.answer_decoder.generate(
            a_init_state, c_ids)

        q_ids = self.question_decoder.generate(q_init_state, c_ids, a_ids)

        return q_ids, start_positions, end_positions

    def return_answer_logits(self, zq, za, c_ids):
        _, a_init_state = self.return_init_state(zq, za)

        start_logits, end_logits = self.answer_decoder(a_init_state, c_ids)

        return start_logits, end_logits

    def get_vae_params(self, lr=1e-3):
        # Get params exclude infomax params
        params = filter(lambda p: p.requires_grad and (
            not hasattr(p, "is_infomax_param")), self.parameters())
        return [{"params": params, "lr": lr}]

    # def get_infomax_params(self, lr=1e-5):
    #     return [{"params": self.answer_infomax_net.parameters(), "lr": lr}]

