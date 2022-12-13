import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.customized_layers import CustomLSTM
from model.model_utils import return_mask_lengths, cal_attn
from torch_scatter import scatter_max
from model.infomax.dim_bce_infomax import DimBceInfoMax

class _ContextEncoderforQG(nn.Module):
    def __init__(self, embedding, emsize,
                 nhidden, nlayers,
                 dropout=0.0):
        super(_ContextEncoderforQG, self).__init__()
        self.embedding = embedding
        self.context_lstm = CustomLSTM(input_size=emsize,
                                       hidden_size=nhidden,
                                       num_layers=nlayers,
                                       dropout=dropout,
                                       bidirectional=True)
        self.context_linear = nn.Linear(2 * nhidden, 2 * nhidden)
        self.fusion = nn.Linear(4 * nhidden, 2 * nhidden, bias=False)
        self.gate = nn.Linear(4 * nhidden, 2 * nhidden, bias=False)

    def forward(self, c_ids, a_ids):
        c_mask, c_lengths = return_mask_lengths(c_ids)

        c_embeddings = self.embedding(c_ids, c_mask, a_ids)
        c_outputs, _ = self.context_lstm(c_embeddings, c_lengths.to("cpu"))
        # attention
        # For attention calculation, linear layer is there for projection
        mask = torch.matmul(c_mask.unsqueeze(2), c_mask.unsqueeze(1))
        c_attned_by_c, _ = cal_attn(self.context_linear(c_outputs),
                                    c_outputs,
                                    mask)
        c_concat = torch.cat([c_outputs, c_attned_by_c], dim=2)
        c_fused = self.fusion(c_concat).tanh()
        c_gate = self.gate(c_concat).sigmoid()
        c_outputs = c_gate * c_fused + (1 - c_gate) * c_outputs
        return c_outputs


class QuestionDecoder(nn.Module):
    def __init__(self, sos_id, eos_id,
                 embedding, contextualized_embedding, emsize,
                 nhidden, ntokens, nlayers,
                 dropout=0.0,
                 max_q_len=64):
        super(QuestionDecoder, self).__init__()

        self.sos_id = sos_id
        self.eos_id = eos_id
        self.emsize = emsize
        self.embedding = embedding
        self.nhidden = nhidden
        self.ntokens = ntokens
        self.nlayers = nlayers
        # this max_len include sos eos
        self.max_q_len = max_q_len

        self.context_lstm = _ContextEncoderforQG(contextualized_embedding, emsize,
                                                nhidden // 2, nlayers, dropout)

        self.question_lstm = CustomLSTM(input_size=emsize,
                                        hidden_size=nhidden,
                                        num_layers=nlayers,
                                        dropout=dropout,
                                        bidirectional=False)

        self.question_linear = nn.Linear(nhidden, nhidden)

        self.concat_linear = nn.Sequential(nn.Linear(2*nhidden, 2*nhidden),
                                           nn.Mish(True),
                                           nn.Dropout(dropout),
                                           nn.Linear(2*nhidden, 2*emsize),
                                           nn.Mish(True))

        self.logit_linear = nn.Linear(emsize, ntokens, bias=False)

        # fix output word matrix
        self.logit_linear.weight = embedding.word_embeddings.weight
        for param in self.logit_linear.parameters():
            param.requires_grad = False

        # self.discriminator = nn.Bilinear(emsize, nhidden, 1)
        self.infomax_est = DimBceInfoMax(emsize, nhidden, use_billinear=True)

    def postprocess(self, q_ids):
        eos_mask = q_ids == self.eos_id
        no_eos_idx_sum = (eos_mask.sum(dim=1) == 0).long() * \
            (self.max_q_len - 1)
        eos_mask = eos_mask.cpu().numpy()
        q_lengths = np.argmax(eos_mask, axis=1) + 1
        q_lengths = torch.tensor(q_lengths).to(
            q_ids.device).long() + no_eos_idx_sum
        batch_size, max_len = q_ids.size()
        idxes = torch.arange(0, max_len).to(q_ids.device)
        idxes = idxes.unsqueeze(0).repeat(batch_size, 1)
        q_mask = (idxes < q_lengths.unsqueeze(1))
        q_ids = q_ids.long() * q_mask.long()
        return q_ids

    def forward(self, init_state, c_ids, q_ids, a_ids):
        batch_size, max_q_len = q_ids.size()

        c_outputs = self.context_lstm(c_ids, a_ids)

        c_mask, _ = return_mask_lengths(c_ids)
        q_mask, q_lengths = return_mask_lengths(q_ids)

        # question dec
        q_embeddings = self.embedding(q_ids)
        q_outputs, _ = self.question_lstm(q_embeddings, q_lengths.to("cpu"), init_state)

        # attention
        # For attention calculation, linear layer is there for projection
        mask = torch.matmul(q_mask.unsqueeze(2), c_mask.unsqueeze(1))
        c_attned_by_q, attn_logits = cal_attn(self.question_linear(q_outputs),
                                              c_outputs,
                                              mask)

        # gen logits
        q_concated = torch.cat([q_outputs, c_attned_by_q], dim=2)
        q_concated = self.concat_linear(q_concated)
        q_maxouted, _ = q_concated.view(
            batch_size, max_q_len, self.emsize, 2).max(dim=-1)
        gen_logits = self.logit_linear(q_maxouted)

        # copy logits
        bq = batch_size * max_q_len
        c_ids = c_ids.unsqueeze(1).repeat(
            1, max_q_len, 1).view(bq, -1).contiguous()
        attn_logits = attn_logits.view(bq, -1).contiguous()
        copy_logits = torch.zeros(bq, self.ntokens).to(c_ids.device)
        copy_logits = copy_logits - 10000.0
        copy_logits, _ = scatter_max(attn_logits, c_ids, out=copy_logits)
        copy_logits = copy_logits.masked_fill(copy_logits == -10000.0, 0)
        copy_logits = copy_logits.view(batch_size, max_q_len, -1).contiguous()

        logits = gen_logits + copy_logits

        if self.training:
            # mutual information btw answer and question (customized: use bi-lstm to average the question & answer)
            a_emb = c_outputs * a_ids.float().unsqueeze(2)
            a_mean_emb = torch.sum(a_emb, 1) / a_ids.sum(1).unsqueeze(1).float()

            q_emb = q_maxouted * q_mask.unsqueeze(2)
            q_mean_emb = torch.sum(q_emb, 1) / q_lengths.unsqueeze(1).float()

            return logits, self.infomax_est(q_mean_emb, a_mean_emb)
        else:
            return logits


    def generate(self, init_state, c_ids, a_ids):
        c_mask, _ = return_mask_lengths(c_ids)
        c_outputs = self.context_lstm(c_ids, a_ids)

        batch_size = c_ids.size(0)

        q_ids = torch.LongTensor([self.sos_id] * batch_size).unsqueeze(1)
        q_ids = q_ids.to(c_ids.device)
        token_type_ids = torch.zeros_like(q_ids)
        position_ids = torch.zeros_like(q_ids)
        q_embeddings = self.embedding(q_ids, token_type_ids, position_ids)

        state = init_state

        # unroll
        all_q_ids = list()
        all_q_ids.append(q_ids)
        for _ in range(self.max_q_len - 1):
            position_ids = position_ids + 1
            q_outputs, state = self.question_lstm.lstm(q_embeddings, state)

            # attention
            mask = c_mask.unsqueeze(1)
            c_attned_by_q, attn_logits = cal_attn(self.question_linear(q_outputs),
                                                  c_outputs,
                                                  mask)

            # gen logits
            q_concated = torch.cat([q_outputs, c_attned_by_q], dim=2)
            q_concated = self.concat_linear(q_concated)
            q_maxouted, _ = q_concated.view(
                batch_size, 1, self.emsize, 2).max(dim=-1)
            gen_logits = self.logit_linear(q_maxouted)

            # copy logits
            attn_logits = attn_logits.squeeze(1)
            copy_logits = torch.zeros(
                batch_size, self.ntokens).to(c_ids.device)
            copy_logits = copy_logits - 10000.0
            copy_logits, _ = scatter_max(attn_logits, c_ids, out=copy_logits)
            copy_logits = copy_logits.masked_fill(copy_logits == -10000.0, 0)

            logits = gen_logits + copy_logits.unsqueeze(1)

            q_ids = torch.argmax(logits, 2)
            all_q_ids.append(q_ids)

            q_embeddings = self.embedding(q_ids, token_type_ids, position_ids)

        q_ids = torch.cat(all_q_ids, 1)
        q_ids = self.postprocess(q_ids)

        return q_ids


    def sample(self, init_state, c_ids, a_ids):
        c_mask, _ = return_mask_lengths(c_ids)
        c_outputs = self.context_lstm(c_ids, a_ids)

        batch_size = c_ids.size(0)

        q_ids = torch.LongTensor([self.sos_id] * batch_size).unsqueeze(1)
        q_ids = q_ids.to(c_ids.device)
        token_type_ids = torch.zeros_like(q_ids)
        position_ids = torch.zeros_like(q_ids)
        q_embeddings = self.embedding(q_ids, token_type_ids, position_ids)

        state = init_state

        # unroll
        all_q_ids = list()
        all_q_ids.append(q_ids)
        for _ in range(self.max_q_len - 1):
            position_ids = position_ids + 1
            q_outputs, state = self.question_lstm.lstm(q_embeddings, state)

            # attention
            mask = c_mask.unsqueeze(1)
            c_attned_by_q, attn_logits = cal_attn(self.question_linear(q_outputs),
                                                  c_outputs,
                                                  mask)

            # gen logits
            q_concated = torch.cat([q_outputs, c_attned_by_q], dim=2)
            q_concated = self.concat_linear(q_concated)
            q_maxouted, _ = q_concated.view(batch_size, 1, self.emsize, 2).max(dim=-1)
            gen_logits = self.logit_linear(q_maxouted)

            # copy logits
            attn_logits = attn_logits.squeeze(1)
            copy_logits = torch.zeros(batch_size, self.ntokens).to(c_ids.device)
            copy_logits = copy_logits - 10000.0
            copy_logits, _ = scatter_max(attn_logits, c_ids, out=copy_logits)
            copy_logits = copy_logits.masked_fill(copy_logits == -10000.0, 0)

            logits = gen_logits + copy_logits.unsqueeze(1)
            logits = logits.squeeze(1)
            logits = self.top_k_top_p_filtering(logits, 2, top_p=0.8)
            probs = F.softmax(logits, dim=-1)
            q_ids = torch.multinomial(probs, num_samples=1)  # [b,1]
            all_q_ids.append(q_ids)

            q_embeddings = self.embedding(q_ids, token_type_ids, position_ids)

        q_ids = torch.cat(all_q_ids, 1)
        q_ids = self.postprocess(q_ids)

        return q_ids