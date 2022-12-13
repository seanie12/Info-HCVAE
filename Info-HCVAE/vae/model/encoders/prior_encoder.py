import torch
import torch.nn as nn
from model.customized_layers import CustomLSTM
from model.model_utils import return_mask_lengths, cal_attn, sample_gaussian, gumbel_softmax

class PriorEncoder(nn.Module):
    def __init__(self, embedding, emsize,
                 nhidden, nlayers,
                 nzqdim, nza, nzadim,
                 dropout=0):
        super(PriorEncoder, self).__init__()

        self.embedding = embedding
        self.nhidden = nhidden
        self.nlayers = nlayers
        self.nzqdim = nzqdim
        self.nza = nza
        self.nzadim = nzadim

        self.context_encoder = CustomLSTM(input_size=emsize,
                                          hidden_size=nhidden,
                                          num_layers=nlayers,
                                          dropout=dropout,
                                          bidirectional=True)

        self.zq_attention = nn.Linear(nzqdim, 2 * nhidden)

        self.zq_linear = nn.Linear(2 * nhidden, 2 * nzqdim)
        self.za_linear = nn.Linear(nzqdim + 2 * 2 * nhidden, nza * nzadim)


    def forward(self, c_ids, zq=None):
        """Produce prior question and answer representation samples

        Args:
            c_ids (torch.Tensor(size=(N, seq_len))): context token ids
            zq (torch.Tensor(size=(N, nzqdim)), optional): Pre-generated question latent variable. 
                Only used for generating answer latent variable for the purpose of evaluation, i.e., 
                only used when self.training=False. Defaults to None.
        """
        c_mask, c_lengths = return_mask_lengths(c_ids)

        c_embeddings = self.embedding(c_ids)
        c_hs, c_state = self.context_encoder(c_embeddings, c_lengths.to("cpu"))
        c_h = c_state[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        c_h = c_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)

        # Sample zq during training and during evaluation only when method param's `zq`=None
        if self.training or (not self.training and zq is None):
            zq_mu, zq_logvar = torch.split(self.zq_linear(c_h), self.nzqdim, dim=1)
            zq = sample_gaussian(zq_mu, zq_logvar)

        # For attention calculation, linear layer is there for projection
        mask = c_mask.unsqueeze(1)
        c_attned_by_zq, _ = cal_attn(self.zq_attention(zq).unsqueeze(1),
                                     c_hs, mask)
        c_attned_by_zq = c_attned_by_zq.squeeze(1)

        h = torch.cat([zq, c_attned_by_zq, c_h], dim=-1)

        za_logits = self.za_linear(h).view(-1, self.nza, self.nzadim)
        # za_prob = F.softmax(za_logits, dim=-1)
        za = gumbel_softmax(za_logits, hard=True)

        if self.training:
            return zq_mu, zq_logvar, zq, za_logits, za
        else:
            return zq, za