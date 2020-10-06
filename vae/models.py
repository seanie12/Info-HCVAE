import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_scatter import scatter_max
from transformers import BertModel, BertTokenizer


def return_mask_lengths(ids):
    mask = torch.sign(ids).float()
    lengths = torch.sum(mask, 1)
    return mask, lengths


def cal_attn(query, memories, mask):
    mask = (1.0 - mask.float()) * -10000.0
    attn_logits = torch.matmul(query, memories.transpose(-1, -2).contiguous())
    attn_logits = attn_logits + mask
    attn_weights = F.softmax(attn_logits, dim=-1)
    attn_outputs = torch.matmul(attn_weights, memories)
    return attn_outputs, attn_logits


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-20, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor

    gumbels = -(torch.empty_like(logits).exponential_() +
                eps).log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Re-parametrization trick.
        ret = y_soft
    return ret


class CategoricalKLLoss(nn.Module):
    def __init__(self):
        super(CategoricalKLLoss, self).__init__()

    def forward(self, P, Q):
        log_P = P.log()
        log_Q = Q.log()
        kl = (P * (log_P - log_Q)).sum(dim=-1).sum(dim=-1)
        return kl.mean(dim=0)


class GaussianKLLoss(nn.Module):
    def __init__(self):
        super(GaussianKLLoss, self).__init__()

    def forward(self, mu1, logvar1, mu2, logvar2):
        numerator = logvar1.exp() + torch.pow(mu1 - mu2, 2)
        fraction = torch.div(numerator, (logvar2.exp()))
        kl = 0.5 * torch.sum(logvar2 - logvar1 + fraction - 1, dim=1)
        return kl.mean(dim=0)


class Embedding(nn.Module):
    def __init__(self, bert_model):
        super(Embedding, self).__init__()
        bert_embeddings = BertModel.from_pretrained(bert_model).embeddings
        self.word_embeddings = bert_embeddings.word_embeddings
        self.token_type_embeddings = bert_embeddings.token_type_embeddings
        self.position_embeddings = bert_embeddings.position_embeddings
        self.LayerNorm = bert_embeddings.LayerNorm
        self.dropout = bert_embeddings.dropout

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if position_ids is None:
            seq_length = input_ids.size(1)
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class ContextualizedEmbedding(nn.Module):
    def __init__(self, bert_model):
        super(ContextualizedEmbedding, self).__init__()
        bert = BertModel.from_pretrained(bert_model)
        self.embedding = bert.embeddings
        self.encoder = bert.encoder
        self.num_hidden_layers = bert.config.num_hidden_layers

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(
            1).unsqueeze(2).float()
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.num_hidden_layers

        embedding_output = self.embedding(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]

        return sequence_output


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional=False):
        super(CustomLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)
        if dropout > 0.0 and num_layers == 1:
            dropout = 0.0

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout,
                            bidirectional=bidirectional, batch_first=True)

    def forward(self, inputs, input_lengths, state=None):
        _, total_length, _ = inputs.size()

        input_packed = pack_padded_sequence(inputs, input_lengths,
                                            batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        output_packed, state = self.lstm(input_packed, state)

        output = pad_packed_sequence(
            output_packed, batch_first=True, total_length=total_length)[0]
        output = self.dropout(output)

        return output, state


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
        q_hs, q_state = self.encoder(q_embeddings, q_lengths)
        q_h = q_state[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        q_h = q_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)

        # context enc
        c_embeddings = self.embedding(c_ids)
        c_hs, c_state = self.encoder(c_embeddings, c_lengths)
        c_h = c_state[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        c_h = c_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)

        # context and answer enc
        c_a_embeddings = self.embedding(c_ids, a_ids, None)
        c_a_hs, c_a_state = self.encoder(c_a_embeddings, c_lengths)
        c_a_h = c_a_state[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        c_a_h = c_a_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)

        # attetion q, c
        mask = c_mask.unsqueeze(1)
        c_attned_by_q, _ = cal_attn(self.question_attention(q_h).unsqueeze(1),
                                    c_hs,
                                    mask)
        c_attned_by_q = c_attned_by_q.squeeze(1)

        # attetion c, q
        mask = q_mask.unsqueeze(1)
        q_attned_by_c, _ = cal_attn(self.context_attention(c_h).unsqueeze(1),
                                    q_hs,
                                    mask)
        q_attned_by_c = q_attned_by_c.squeeze(1)

        h = torch.cat([q_h, q_attned_by_c, c_h, c_attned_by_q], dim=-1)

        zq_mu, zq_logvar = torch.split(self.zq_linear(h), self.nzqdim, dim=1)
        zq = zq_mu + torch.randn_like(zq_mu) * torch.exp(0.5 * zq_logvar)

        # attention zq, c_a
        mask = c_mask.unsqueeze(1)
        c_a_attned_by_zq, _ = cal_attn(self.zq_attention(zq).unsqueeze(1),
                                       c_a_hs,
                                       mask)
        c_a_attned_by_zq = c_a_attned_by_zq.squeeze(1)

        h = torch.cat([zq, c_a_attned_by_zq, c_a_h], dim=-1)

        za_logits = self.za_linear(h).view(-1, self.nza, self.nzadim)
        za_prob = F.softmax(za_logits, dim=-1)
        za = gumbel_softmax(za_logits, hard=True)

        return zq_mu, zq_logvar, zq, za_prob, za


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

    def forward(self, c_ids):
        c_mask, c_lengths = return_mask_lengths(c_ids)

        c_embeddings = self.embedding(c_ids)
        c_hs, c_state = self.context_encoder(c_embeddings, c_lengths)
        c_h = c_state[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        c_h = c_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)

        zq_mu, zq_logvar = torch.split(self.zq_linear(c_h), self.nzqdim, dim=1)
        zq = zq_mu + torch.randn_like(zq_mu)*torch.exp(0.5*zq_logvar)

        mask = c_mask.unsqueeze(1)
        c_attned_by_zq, _ = cal_attn(self.zq_attention(zq).unsqueeze(1),
                                     c_hs,
                                     mask)
        c_attned_by_zq = c_attned_by_zq.squeeze(1)

        h = torch.cat([zq, c_attned_by_zq, c_h], dim=-1)

        za_logits = self.za_linear(h).view(-1, self.nza, self.nzadim)
        za_prob = F.softmax(za_logits, dim=-1)
        za = gumbel_softmax(za_logits, hard=True)

        return zq_mu, zq_logvar, zq, za_prob, za

    def interpolation(self, c_ids, zq):

        c_mask, c_lengths = return_mask_lengths(c_ids)

        c_embeddings = self.embedding(c_ids)
        c_hs, c_state = self.context_encoder(c_embeddings, c_lengths)
        c_h = c_state[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        c_h = c_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)

        mask = c_mask.unsqueeze(1)
        c_attned_by_zq, _ = cal_attn(
            self.zq_attention(zq).unsqueeze(1), c_hs, mask)
        c_attned_by_zq = c_attned_by_zq.squeeze(1)

        h = torch.cat([zq, c_attned_by_zq, c_h], dim=-1)

        za_logits = self.za_linear(h).view(-1, self.nza, self.nzadim)
        za = gumbel_softmax(za_logits, hard=True)

        return za


class AnswerDecoder(nn.Module):
    def __init__(self, embedding, emsize,
                 nhidden, nlayers,
                 dropout=0.0):
        super(AnswerDecoder, self).__init__()

        self.embedding = embedding

        self.context_lstm = CustomLSTM(input_size=4 * emsize,
                                       hidden_size=nhidden,
                                       num_layers=nlayers,
                                       dropout=dropout,
                                       bidirectional=True)

        self.start_linear = nn.Linear(2 * nhidden, 1)
        self.end_linear = nn.Linear(2 * nhidden, 1)
        self.ls = nn.LogSoftmax(dim=1)

    def forward(self, init_state, c_ids):
        _, max_c_len = c_ids.size()
        c_mask, c_lengths = return_mask_lengths(c_ids)

        H = self.embedding(c_ids, c_mask)
        U = init_state.unsqueeze(1).repeat(1, max_c_len, 1)
        G = torch.cat([H, U, H * U, torch.abs(H - U)], dim=-1)
        M, _ = self.context_lstm(G, c_lengths)

        start_logits = self.start_linear(M).squeeze(-1)
        end_logits = self.end_linear(M).squeeze(-1)

        start_end_mask = (c_mask == 0)
        masked_start_logits = start_logits.masked_fill(
            start_end_mask, -10000.0)
        masked_end_logits = end_logits.masked_fill(start_end_mask, -10000.0)

        return masked_start_logits, masked_end_logits

    def generate(self, init_state, c_ids):
        start_logits, end_logits = self.forward(init_state, c_ids)
        c_mask, _ = return_mask_lengths(c_ids)
        batch_size, max_c_len = c_ids.size()

        mask = torch.matmul(c_mask.unsqueeze(2).float(),
                            c_mask.unsqueeze(1).float())
        mask = torch.triu(mask) == 0
        score = (self.ls(start_logits).unsqueeze(2)
                 + self.ls(end_logits).unsqueeze(1))
        score = score.masked_fill(mask, -10000.0)
        score, start_positions = score.max(dim=1)
        score, end_positions = score.max(dim=1)
        start_positions = torch.gather(start_positions,
                                       1,
                                       end_positions.view(-1, 1)).squeeze(1)

        idxes = torch.arange(0, max_c_len, out=torch.LongTensor(max_c_len))
        idxes = idxes.unsqueeze(0).to(
            start_logits.device).repeat(batch_size, 1)

        start_positions = start_positions.unsqueeze(1)
        start_mask = (idxes >= start_positions).long()
        end_positions = end_positions.unsqueeze(1)
        end_mask = (idxes <= end_positions).long()
        a_ids = start_mask + end_mask - 1

        return a_ids, start_positions.squeeze(1), end_positions.squeeze(1)


class ContextEncoderforQG(nn.Module):
    def __init__(self, embedding, emsize,
                 nhidden, nlayers,
                 dropout=0.0):
        super(ContextEncoderforQG, self).__init__()
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
        c_outputs, _ = self.context_lstm(c_embeddings, c_lengths)
        # attention
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

        self.context_lstm = ContextEncoderforQG(contextualized_embedding, emsize,
                                                nhidden // 2, nlayers, dropout)

        self.question_lstm = CustomLSTM(input_size=emsize,
                                        hidden_size=nhidden,
                                        num_layers=nlayers,
                                        dropout=dropout,
                                        bidirectional=False)

        self.question_linear = nn.Linear(nhidden, nhidden)

        self.concat_linear = nn.Sequential(nn.Linear(2*nhidden, 2*nhidden),
                                           nn.Dropout(dropout),
                                           nn.Linear(2*nhidden, 2*emsize))

        self.logit_linear = nn.Linear(emsize, ntokens, bias=False)

        # fix output word matrix
        self.logit_linear.weight = embedding.word_embeddings.weight
        for param in self.logit_linear.parameters():
            param.requires_grad = False

        self.discriminator = nn.Bilinear(emsize, nhidden, 1)

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
        q_outputs, _ = self.question_lstm(q_embeddings, q_lengths, init_state)

        # attention
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

        # mutual information btw answer and question
        a_emb = c_outputs * a_ids.float().unsqueeze(2)
        a_mean_emb = torch.sum(a_emb, 1) / a_ids.sum(1).unsqueeze(1).float()
        fake_a_mean_emb = torch.cat([a_mean_emb[-1].unsqueeze(0),
                                     a_mean_emb[:-1]], dim=0)

        q_emb = q_maxouted * q_mask.unsqueeze(2)
        q_mean_emb = torch.sum(q_emb, 1) / q_lengths.unsqueeze(1).float()
        fake_q_mean_emb = torch.cat([q_mean_emb[-1].unsqueeze(0),
                                     q_mean_emb[:-1]], dim=0)

        bce_loss = nn.BCEWithLogitsLoss()
        true_logits = self.discriminator(q_mean_emb, a_mean_emb)
        true_labels = torch.ones_like(true_logits)

        fake_a_logits = self.discriminator(q_mean_emb, fake_a_mean_emb)
        fake_q_logits = self.discriminator(fake_q_mean_emb, a_mean_emb)
        fake_logits = torch.cat([fake_a_logits, fake_q_logits], dim=0)
        fake_labels = torch.zeros_like(fake_logits)

        true_loss = bce_loss(true_logits, true_labels)
        fake_loss = 0.5 * bce_loss(fake_logits, fake_labels)
        loss_info = 0.5 * (true_loss + fake_loss)

        return logits, loss_info

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
        c_mask, c_lengths = return_mask_lengths(c_ids)
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


class DiscreteVAE(nn.Module):
    def __init__(self, args):
        super(DiscreteVAE, self).__init__()
        tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        padding_idx = tokenizer.vocab['[PAD]']
        sos_id = tokenizer.vocab['[CLS]']
        eos_id = tokenizer.vocab['[SEP]']
        ntokens = len(tokenizer.vocab)

        bert_model = args.bert_model
        if "large" in bert_model:
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

        self.lambda_kl = args.lambda_kl
        self.lambda_info = args.lambda_info

        max_q_len = args.max_q_len

        embedding = Embedding(bert_model)
        contextualized_embedding = ContextualizedEmbedding(bert_model)
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
                                                dec_q_dropout,
                                                max_q_len)

        self.q_h_linear = nn.Linear(nzqdim, dec_q_nlayers * dec_q_nhidden)
        self.q_c_linear = nn.Linear(nzqdim, dec_q_nlayers * dec_q_nhidden)
        self.a_linear = nn.Linear(nza * nzadim, emsize, False)

        self.q_rec_criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
        self.gaussian_kl_criterion = GaussianKLLoss()
        self.categorical_kl_criterion = CategoricalKLLoss()

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
            posterior_za_prob, posterior_za \
            = self.posterior_encoder(c_ids, q_ids, a_ids)

        prior_zq_mu, prior_zq_logvar, _, \
            prior_za_prob, _ \
            = self.prior_encoder(c_ids)

        q_init_state, a_init_state = self.return_init_state(
            posterior_zq, posterior_za)

        # answer decoding
        start_logits, end_logits = self.answer_decoder(a_init_state, c_ids)
        # question decoding
        q_logits, loss_info = self.question_decoder(
            q_init_state, c_ids, q_ids, a_ids)

        # q rec loss
        loss_q_rec = self.q_rec_criterion(q_logits[:, :-1, :].transpose(1, 2).contiguous(),
                                          q_ids[:, 1:])

        # a rec loss
        max_c_len = c_ids.size(1)
        a_rec_criterion = nn.CrossEntropyLoss(ignore_index=max_c_len)
        start_positions.clamp_(0, max_c_len)
        end_positions.clamp_(0, max_c_len)
        loss_start_a_rec = a_rec_criterion(start_logits, start_positions)
        loss_end_a_rec = a_rec_criterion(end_logits, end_positions)
        loss_a_rec = 0.5 * (loss_start_a_rec + loss_end_a_rec)

        # kl loss
        loss_zq_kl = self.gaussian_kl_criterion(posterior_zq_mu,
                                                posterior_zq_logvar,
                                                prior_zq_mu,
                                                prior_zq_logvar)

        loss_za_kl = self.categorical_kl_criterion(posterior_za_prob,
                                                   prior_za_prob)

        loss_kl = self.lambda_kl * (loss_zq_kl + loss_za_kl)
        loss_info = self.lambda_info * loss_info

        loss = loss_q_rec + loss_a_rec + loss_kl + loss_info

        return loss, \
            loss_q_rec, loss_a_rec, \
            loss_zq_kl, loss_za_kl, \
            loss_info

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
