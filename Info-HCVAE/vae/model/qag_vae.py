import torch
import torch.nn as nn
from transformers import BertTokenizer
from model.customized_layers import ContextualizedEmbedding, Embedding
from model.encoders import PosteriorEncoder, PriorEncoder
from model.decoders import QuestionDecoder, AnswerDecoder
from model.losses import GaussianKLLoss, CategoricalKLLoss, ContinuousKernelMMDLoss, CategoricalMMDLoss, DIoUAnswerSpanLoss
from model.infomax import InfoMaxModel

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

        self.w_ans = args.w_ans
        self.w_bce = args.w_bce
        self.alpha_kl = args.alpha_kl
        self.lambda_mmd = args.lambda_mmd
        self.lambda_z_info = args.lambda_z_info
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
        self.a_linear = nn.Linear(nza*nzadim, emsize, False)

        self.q_rec_criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
        self.gaussian_kl_criterion = GaussianKLLoss()
        self.categorical_kl_criterion = CategoricalKLLoss()

        if self.alpha_kl + self.lambda_mmd - 1 > 0:
            self.continuous_mmd_criterion = ContinuousKernelMMDLoss()
            self.categorical_mmd_criterion = CategoricalMMDLoss()

        if self.lambda_z_info > 0:
            self.posterior_zq_info = InfoMaxModel(x_dim=2*emsize, z_dim=nzqdim)
            # self.posterior_zq_info.denote_infomax_net_for_params()
            # self.prior_zq_info = InfoMaxModel(x_dim=emsize, z_dim=nzqdim)
            # self.prior_zq_info.denote_infomax_net_for_params()

            self.posterior_za_info = InfoMaxModel(x_dim=2*emsize, z_dim=nza*nzadim)
            # self.posterior_za_info.denote_infomax_net_for_params()
            # self.prior_za_info = InfoMaxModel(x_dim=emsize, z_dim=nza*nzadim)
            # self.prior_za_info.denote_infomax_net_for_params()


    def return_init_state(self, zq, za):
        q_init_h = self.q_h_linear(zq)
        q_init_c = self.q_c_linear(zq)
        q_init_h = q_init_h.view(-1, self.dec_q_nlayers,
                                 self.dec_q_nhidden).transpose(0, 1).contiguous()
        q_init_c = q_init_c.view(-1, self.dec_q_nlayers,
                                 self.dec_q_nhidden).transpose(0, 1).contiguous()
        q_init_state = (q_init_h, q_init_c)

        za_flatten = za.view(-1, self.nza * self.nzadim)
        a_init_state = self.a_linear(za_flatten)

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
        start_logits, end_logits = self.answer_decoder(a_init_state, c_ids)
        # question decoding
        q_logits, loss_info = self.question_decoder(q_init_state, c_ids, q_ids, a_ids)

        # Compute losses
        if self.training:
            # q rec loss
            loss_q_rec = self.q_rec_criterion(q_logits[:, :-1, :].transpose(1, 2).contiguous(),
                                            q_ids[:, 1:])

            # a rec loss
            max_c_len = c_ids.size(1)
            # Emphasize the importance of predicting the correct span
            a_rec_criterion = nn.CrossEntropyLoss(ignore_index=max_c_len)
            start_positions.clamp_(0, max_c_len)
            end_positions.clamp_(0, max_c_len)
            loss_start_a_rec = a_rec_criterion(start_logits, start_positions)
            loss_end_a_rec = a_rec_criterion(end_logits, end_positions)
            loss_a_rec = self.w_ans * 0.5 * (loss_start_a_rec + loss_end_a_rec)

            # kl loss
            loss_zq_kl = self.gaussian_kl_criterion(posterior_zq_mu, posterior_zq_logvar,
                                            prior_zq_mu, prior_zq_logvar)

            loss_za_kl = self.w_ans * self.categorical_kl_criterion(posterior_za_logits,
                                                        prior_za_logits)

            loss_zq_mmd, loss_za_mmd = torch.tensor(0), torch.tensor(0)
            if self.alpha_kl + self.lambda_mmd - 1 > 0:
                loss_zq_mmd = self.continuous_mmd_criterion(posterior_zq_mu, posterior_zq_logvar,
                                            prior_zq_mu, prior_zq_logvar)
                loss_za_mmd = self.w_ans * self.categorical_mmd_criterion(posterior_za_logits, prior_za_logits)

            loss_zq_info, loss_za_info = torch.tensor(0), torch.tensor(0)
            if self.lambda_z_info > 0:
                mean_c_embs = self.posterior_encoder.embedding(c_ids).mean(dim=1)
                mean_q_embs = self.posterior_encoder.embedding(q_ids).mean(dim=1)
                mean_c_a_embs = self.posterior_encoder.embedding(c_ids, a_ids, None).mean(dim=1)
                # soft_posterior_za = sample_gumbel(posterior_za_logits, hard=False).view(-1, self.nza*self.nzadim)
                # soft_prior_za = sample_gumbel(prior_za_logits, hard=False).view(-1, self.nza*self.nzadim)

                # loss_zq_info = self.posterior_zq_info(torch.cat([mean_c_embs, mean_q_embs], dim=-1), posterior_zq) \
                #             + self.prior_zq_info(mean_c_embs, prior_zq)
                # loss_za_info = self.posterior_za_info(mean_c_a_embs, posterior_za.view(-1, self.nza*self.nzadim)) \
                #             + self.prior_za_info(mean_c_embs, prior_za.view(-1, self.nza*self.nzadim))
                loss_zq_info = self.posterior_zq_info(torch.cat([mean_c_embs, mean_q_embs], dim=-1), posterior_zq)
                loss_za_info = self.posterior_za_info(torch.cat([mean_q_embs, mean_c_a_embs], dim=-1), posterior_za.view(-1, self.nza*self.nzadim))
                # use exp to prevent the infomax to become too negative
                loss_zq_info = torch.exp(loss_zq_info)
                loss_za_info = torch.exp(loss_za_info)

            loss_kl = (1.0 - self.alpha_kl) * (loss_zq_kl + loss_za_kl)
            loss_mmd = (self.alpha_kl + self.lambda_mmd - 1) * (loss_zq_mmd + loss_za_mmd)
            loss_z_info = self.lambda_z_info * (loss_zq_info + loss_za_info)
            loss_qa_info = self.lambda_qa_info * loss_info

            loss = self.w_bce * (loss_q_rec + loss_a_rec) + loss_kl + loss_qa_info + loss_mmd + loss_z_info

            return_dict = {
                "total_loss": loss,
                "loss_q_rec": loss_q_rec,
                "loss_a_rec": loss_a_rec,
                "loss_kl": loss_kl,
                "loss_zq_kl": loss_zq_kl,
                "loss_za_kl": loss_za_kl,
                "loss_mmd": loss_mmd,
                "loss_zq_mmd": loss_zq_mmd,
                "loss_za_mmd": loss_za_mmd,
                "loss_z_info": loss_z_info,
                "loss_zq_info": loss_zq_info,
                "loss_za_info": loss_za_info,
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
        params = filter(lambda p: p.requires_grad and (not hasattr(p, "is_infomax_param")), self.parameters())
        return [ { "params": params, "lr": lr } ]


    def get_infomax_params(self, lr=1e-5):
        return [ { "params": self.posterior_zq_info.parameters(), "lr": lr },
            { "params": self.prior_zq_info.parameters(), "lr": lr },
            { "params": self.posterior_za_info.parameters(), "lr": lr },
            { "params": self.prior_za_info.parameters(), "lr": lr } ]