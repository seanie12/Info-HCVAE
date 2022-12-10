import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel


class Embedding(nn.Module):
    def __init__(self, huggingface_model):
        super(Embedding, self).__init__()
        self.transformer_embeddings = BertModel.from_pretrained(
            huggingface_model).embeddings
        self.word_embeddings = self.transformer_embeddings.word_embeddings
        self.token_type_embeddings = self.transformer_embeddings.token_type_embeddings
        self.position_embeddings = self.transformer_embeddings.position_embeddings
        self.LayerNorm = self.transformer_embeddings.LayerNorm
        self.dropout = self.transformer_embeddings.dropout

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
    def __init__(self, huggingface_model):
        super(ContextualizedEmbedding, self).__init__()
        bert = BertModel.from_pretrained(huggingface_model)
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
