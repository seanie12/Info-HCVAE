import torch
import torch.nn as nn
from model.customized_layers import CustomLSTM
from model.model_utils import return_mask_lengths, cal_attn, gumbel_softmax, sample_gaussian

class PosteriorEncoder(nn.Module):
    def __init__(self, embedding, emsize,
                 nhidden, nlayers,
                 nzqdim, nza, nzadim,
                 dropout=0.0):
        super(PosteriorEncoder, self).__init__()

        self.embedding = embedding
        self.nhidden = nhidden
        self.nlayers = nlayers
        self.nzqdim = nzqdim
        self.nza = nza
        self.nzadim = nzadim

        self.encoder = CustomLSTM(input_size=emsize,
                                  hidden_size=nhidden,
                                  num_layers=nlayers,
                                  dropout=dropout,
                                  bidirectional=True)

        self.question_attention = nn.Linear(2 * nhidden, 2 * nhidden)
        self.context_attention = nn.Linear(2 * nhidden, 2 * nhidden)
        self.zq_attention = nn.Linear(nzqdim, 2 * nhidden)

        self.zq_linear = nn.Linear(4 * 2 * nhidden, 2 * nzqdim)
        self.za_linear = nn.Linear(nzqdim + 2 * 2 * nhidden, nza * nzadim)

    def forward(self, c_ids, q_ids, a_ids):
        c_mask, c_lengths = return_mask_lengths(c_ids)
        q_mask, q_lengths = return_mask_lengths(q_ids)

        # question enc
        q_embeddings = self.embedding(q_ids)
        q_hs, q_state = self.encoder(q_embeddings, q_lengths.to("cpu"))
        q_h = q_state[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        q_h = q_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)

        # context enc
        c_embeddings = self.embedding(c_ids)
        c_hs, c_state = self.encoder(c_embeddings, c_lengths.to("cpu"))
        c_h = c_state[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        c_h = c_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)

        # context and answer enc
        c_a_embeddings = self.embedding(c_ids, a_ids, None)
        c_a_hs, c_a_state = self.encoder(c_a_embeddings, c_lengths.to("cpu"))
        c_a_h = c_a_state[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        c_a_h = c_a_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)

        # attetion q, c
        # For attention calculation, linear layer is there for projection
        mask = c_mask.unsqueeze(1)
        c_attned_by_q, _ = cal_attn(self.question_attention(q_h).unsqueeze(1),
                                    c_hs,
                                    mask)
        c_attned_by_q = c_attned_by_q.squeeze(1)

        # attetion c, q
        # For attention calculation, linear layer is there for projection
        mask = q_mask.unsqueeze(1)
        q_attned_by_c, _ = cal_attn(self.context_attention(c_h).unsqueeze(1),
                                    q_hs,
                                    mask)
        q_attned_by_c = q_attned_by_c.squeeze(1)

        h = torch.cat([q_h, q_attned_by_c, c_h, c_attned_by_q], dim=-1)

        zq_mu, zq_logvar = torch.split(self.zq_linear(h), self.nzqdim, dim=1)
        zq = sample_gaussian(zq_mu, zq_logvar)

        # attention zq, c_a
        # For attention calculation, linear layer is there for projection
        mask = c_mask.unsqueeze(1)
        c_a_attned_by_zq, _ = cal_attn(self.zq_attention(zq).unsqueeze(1),
                                       c_a_hs,
                                       mask)
        c_a_attned_by_zq = c_a_attned_by_zq.squeeze(1)

        h = torch.cat([zq, c_a_attned_by_zq, c_a_h], dim=-1)

        za_logits = self.za_linear(h).view(-1, self.nza, self.nzadim)
        # za_prob = F.softmax(za_logits, dim=-1)
        za = gumbel_softmax(za_logits, hard=True)

        if self.training:
            return zq_mu, zq_logvar, zq, za_logits, za
        else:
            return zq, za