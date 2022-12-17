import torch
import torch.nn as nn
from infohcvae.model.customized_layers import CustomLSTM
from infohcvae.model.model_utils import return_mask_lengths


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

        c_embeddings = self.embedding(c_ids, c_mask)
        repeated_init_state = init_state.unsqueeze(1).repeat(1, max_c_len, 1)
        # reshaped_init_state = init_state.view(batch_size, max_c_len, -1)
        fused_features = torch.cat([c_embeddings,
                                    repeated_init_state,
                                    c_embeddings * repeated_init_state,
                                    torch.abs(c_embeddings - repeated_init_state)],
                                   dim=-1)
        out_features, _ = self.context_lstm(
            fused_features, c_lengths.to("cpu"))

        start_logits = self.start_linear(out_features).squeeze(-1)
        end_logits = self.end_linear(out_features).squeeze(-1)

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
