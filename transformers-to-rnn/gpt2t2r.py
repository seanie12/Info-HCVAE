# my code for https://arxiv.org/pdf/2103.13076.pdf
# GNU GPLv3 - @yashbonde

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from transformers.pytorch_utils import Conv1D
from transformers.models.gpt2.modeling_gpt import GPT2MLP
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

##
# Generator class for generation-based models
##
class GenerationMixin():
    @staticmethod
    def top_k_logits(logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -1e10
        return out

    def get_next_tokens(self, logits, top_k, do_sample, num_return_sequences):
        # get probabilities
        if top_k is not None:
            logits = self.top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)

        # get top_k tokens for each
        if do_sample:
            ix = torch.multinomial(probs, num_samples=num_return_sequences)
        else:
            _, ix = torch.topk(probs, k=num_return_sequences, dim=-1)
        return ix

    def get_top_tokens_by_score(self, ix):
        # this manages the tokens to keep based on the scores
        return ix[0]

    # test vanilla generation
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_length,
        num_return_sequences=10,
        top_k=10,
        do_sample=True
    ):
        assert self.infer_ready_flag, "Not ready for inference. see `T2RInfer.infer_init()`"

        # if just a sequence, batchify
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)

        # since the hidden states for all `num_return_sequences` will be same
        # we first run those and then tile
        B, S = input_ids.shape

        assert B == 1, "Only 1 sequence at a time can be generated"

        s_rnn, z_rnn = self.init_hidden()
        B, S = input_ids.shape
        for i in range(S):
            logits, s_rnn, z_rnn = self(
                input_ids[:, i].unsqueeze(0), s_rnn=s_rnn, z_rnn=z_rnn, i=i)

        ix = self.get_next_tokens(
            logits[:, -1, :], top_k, do_sample, num_return_sequences)
        generated_tokens = [ix[0]]

        if num_return_sequences > 1:
            # tile rnn_states for multiple batch generation
            s_rnn = [torch.tile(s, [num_return_sequences, 1, 1])
                     for s in s_rnn]
            z_rnn = [torch.tile(z, [num_return_sequences, 1, 1])
                     for z in z_rnn]

        for i in range(S, max_length - 1, 1):
            logits, s_rnn, z_rnn = self(
                ix.view(num_return_sequences, 1), s_rnn=s_rnn, z_rnn=z_rnn, i=i)
            ix = self.get_next_tokens(
                logits[:, -1, :], top_k, do_sample, num_return_sequences)
            ix = self.get_top_tokens_by_score(ix)
            generated_tokens.append(ix)

        generated_tokens = [x.unsqueeze(-1) for x in generated_tokens]
        generated_tokens = torch.cat(generated_tokens, dim=1)
        input_ids = torch.tile(input_ids, [num_return_sequences, 1])
        full_seq = torch.cat([input_ids, generated_tokens], dim=1)
        return full_seq


class RNNAttention(nn.Module):
    def __init__(self, config):
        """this is at the heart of T2R framework. Auto manages forward() method for
        either standard transformer style full attention or T2R style RNN State Attention.
        A single function that manages this is relevant because then you can merge
        different attention patterns in the single model without changing the code.

        Args:
          config (GPT2Config): configuration for this module
        """
        super().__init__()
        self.config = config
        nx = config.n_embd

        self.c_attn = Conv1D(3 * nx, nx)
        self.c_proj = Conv1D(nx, nx)

        # phi has different weights for different heads and thus
        # W = r*k * d ==> e
        nx_rnn = config.feature_size * config.n_head
        self.phi = nn.Sequential(
            nn.Linear(nx, nx_rnn),
            nn.ReLU()
        )
        self.infer_ready_flag = False

    # functions to replicate standard attention model
    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.config.n_head,
                                       x.size(-1) // self.config.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            # (batch, head, head_features, seq_length)
            return x.permute(0, 2, 3, 1)
        else:
            # (batch, head, seq_length, head_features)
            return x.permute(0, 2, 1, 3)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def standard_forward(self, x, attention_mask):
        # pass through the feature transformer and then pass k,q through
        # phi then split. Since phi has to be different for each head,
        # we can do a large phi and split after that.
        q, k, v = self.c_attn(x).split(self.config.n_embd, dim=2)
        phi_q = self.split_heads(self.phi(q))  # [B,H,S,E//H]
        phi_k = self.split_heads(self.phi(k), k=True)  # [B,H,E//H,S]
        v = self.split_heads(v)
        w = phi_q @ phi_k  # [B,H,S,S]
        if attention_mask is not None:
            w = w + attention_mask
        out = w @ v  # [B,H,S,E//H]
        out = self.merge_heads(out)  # [B,S,E]
        out = self.c_proj(out)
        return [out]

    # functions to behave like RNN
    def infer_init(self):
        """this function performs the merging of weights + refining the graph"""
        config = self.config

        # we get the matrices for the three qkv's by splitting the weights
        # since we anyways split the output of qkv into q,k,v the weight and bias
        # matrix can also be split
        wq, wk, wv = self.c_attn.weight.data.split(config.n_embd, dim=1)
        bq, bk, bv = self.c_attn.bias.data.split(config.n_embd, dim=0)

        # get the matrices for phi
        wp = self.phi[0].weight.data  # [nx, k * n_head]
        bp = self.phi[0].bias.data  # [k * n_head]

        # cache the phi function weights and biases
        wpq = wp @ wq      # [n_head * k, n_embd]
        bpq = bp + wp @ bq  # [n_head * k]
        wpk = wp @ wk      # [n_head * k, n_embd]
        bpk = bp + wp @ bk  # [n_head * k]

        # define standard v layer
        self.v_layer = lambda x: x @ wv + bv

        # difference from paper, division by 0 causes nan error so we add
        # a tiny positive to ensure that does not happen
        self.phi_q = lambda x: torch.relu(x @ wpq.T + bpq) + 0.0001
        self.phi_k = lambda x: torch.relu(x @ wpk.T + bpk) + 0.0001

        # set flag that optimisation for inference is complete
        self.infer_ready_flag = True

    def rnn_forward(self, x, s, z):
        # get q,k,v
        q = self.split_heads(self.phi_q(x))  # [B,H,1,k]
        k = self.split_heads(self.phi_k(x), k=True)  # [B,H,k,1]
        v = self.split_heads(self.v_layer(x))  # [B,H,1,E//H]

        # from previous state we get S and z for t-1
        if not isinstance(s, (int, float)):
            # when initiating we can just send [0, 0]
            s = self.split_heads(s)
            z = self.split_heads(z)
        s = s + k @ v  # [B,H,k,E//H]
        z = z + k  # [B,H,k,1]

        # get output
        num = q @ s  # [B, H, 1, d]
        dem = q @ z  # [B, H, 1, 1]
        out = num * (dem ** -1)  # mults are faster than /

        # pass through transform layer
        out = self.merge_heads(out)  # [B,S,E]
        s = self.merge_heads(s)  # [B,k,E]
        z = self.merge_heads(z)  # [B,k,H]
        out = self.c_proj(out)
        return [out, s, z]

    # generic forward function that auto manages which style of processing to perform
    def forward(self, x, attention_mask=None, s=None, z=None):
        if self.infer_ready_flag:
            assert x.shape[
                1] == 1, f"During inference only one token is allowed got: {x.shape}"
            output = self.rnn_forward(x, s, z)  # out, s, z
        else:
            output = self.standard_forward(x, attention_mask)  # out
        return output


class GPT2RnnBlock(nn.Module):
    def __init__(self, config):
        """gpt2.modeling_gpt2.GPT2RnnBlock modified for use with RNNs

        Args:
          config (GPT2Config): configuration for this module
        """
        super().__init__()

        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = RNNAttention(config)
        # self.attn = gpt2.modeling_gpt2.Attention(config.n_embd, config.n_ctx, config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)

    def infer_init(self):
        self.attn.infer_init()

    def forward(self, hidden_states, attention_mask=None, s=None, z=None):
        # forward pass through attention layer and break into components based on
        # type of method
        attn_outputs = self.attn(
            x=self.ln_1(hidden_states),
            attention_mask=attention_mask,
            s=s, z=z
        )
        if len(attn_outputs) == 1:
            # this is standard transformer feedforward
            attn_outputs = attn_outputs[0]
            s, z = None, None
        elif len(attn_outputs) == 3:
            # this is RNN transformer output
            attn_outputs, s, z, = attn_outputs

        # NOTE: this operation is not correct and will cause errors during loss.backward()
        # hidden_states += attn_outputs
        # because "+=" causes inplace update and not create a new variable meaning that
        # nodes from the graph will be missing.
        hidden_states = hidden_states + attn_outputs
        mlp_states = self.mlp(self.ln_2(hidden_states))
        hidden_states = hidden_states + mlp_states

        # always return these three upstream model will have to manage this
        return hidden_states, s, z


class GPT2T2R(nn.Module, GenerationMixin):
    def __init__(self, config):
        """This is training only module for T2R model and behaves just like a GPT2LMHeadModel.
        You can directly load weights of any ``transformers.GPT2LMHeadModel`` ie. [`gpt`, ...].
        It is loaded with ``GenerationMixin`` a simple generation mixer that can be used to
        generate things oob.

        :Example:

        (Recommended) load from pretrained huggingface models like:

        >>> model = T2RTraining.from_pretrained("gpt2", feature_size = 32)

        Or explicitly load from some model:

        >>> base_model = GPT2LMHeadModel.from_pretrained("gpt2")
        >>> t2r_train = T2RTraining(base_model.config)
        >>> t2r_train.load_state_dict(base_model.state_dict())
        ... <Required Keys Matched>

        Args:
          config (GPT2Config): configuration for this module
        """
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2RnnBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.infer_ready_flag = False

    # methods to make life easier with huggingface
    @classmethod
    def from_pretrained(cls, name: str, feature_size: int, return_tokenizer=False):
        """load from any pretrained huggingface GPT2 model.

        Args:
            name (str): key for pretrained model
            feature_size (int): feature size to use for RNNAttention
        """
        assert "gpt" in name, "Only GPT models are supported"
        base_model = GPT2LMHeadModel.from_pretrained(name)
        config = GPT2Config(**base_model.config.__dict__,
                            feature_size=feature_size)
        new_class = cls(config)
        new_class.load_state_dict(base_model.state_dict())
        del base_model
        tokenizer = GPT2Tokenizer.from_pretrained(name)
        if not return_tokenizer:
            return new_class
        else:
            return new_class, tokenizer

    # methods to make life easier with torch
    @property
    def num_params(self):
        if not self.infer_ready_flag:
            return sum(p.numel() for p in self.parameters())
        else:
            # this is a bit different because we fuse the layers in attention [WIP]
            pass

    def load_state_dict(self, state_dict, strict=False):
        # there is no self.transformer so remove any such reference then check for
        # unexpected layers.
        new_state_dict = []
        for k in state_dict:
            new_state_dict.append((
                k.replace("transformer.", ""),
                state_dict[k]
            ))
        new_state_dict = OrderedDict(new_state_dict)
        incompatiable_keys, unexpected_keys = super(
        ).load_state_dict(new_state_dict, strict)
        incompatiable_keys = [x for x in incompatiable_keys if "phi" not in x]
        unexpected_keys = [
            x for x in unexpected_keys
            if ".".join(x.split(".")[2:]) not in ["attn.masked_bias", "attn.bias"]
        ]
        assert len(
            incompatiable_keys) == 0, f"Found incompatible keys {incompatiable_keys}"
        assert len(
            unexpected_keys) == 0, f"Found unexpected keys {unexpected_keys}"

        return "<Required Keys Matched>"

    # functions for stansdard transformer style forward
    def standard_forward(self, input_ids, attention_mask=None, labels=None):
        # forward works just like GPT2LMHeadModel
        B, S = input_ids.shape

        # Attention mask.
        if attention_mask is not None:
            assert B > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(B, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(
                dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # get embeddings
        device = input_ids.device
        position_ids = torch.arange(
            0, input_ids.shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_ids.shape[-1])
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        # pass through GPT2Rnnblocks
        for i, GPT2Rnnblock in enumerate(self.h):
            # GPT2Rnnblock always returns 3 items
            hidden_states, _, _ = GPT2Rnnblock(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                s=None, z=None
            )

        # get final predictions
        output = self.ln_f(hidden_states)
        logits = self.lm_head(output)
        output = [logits]

        if labels is not None:
            # Shift so that tokens < n predict n & Flatten the tokens
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            output = [loss, logits]  # HF trainer expects loss as first item

        return output

    # functions for RNN style inference
    def infer_init(self):
        """Prepares the model for inference by fusing parameters in RNNAttention Layer
        """
        print("Preparing Model for inference")
        for GPT2Rnnblock in self.h:
            GPT2Rnnblock.infer_init()
        self.infer_ready_flag = True

    def init_hidden(self):
        # create dummy hidden states that cn be used for inference
        s = [
            torch.zeros(size=(1, self.config.feature_size, self.config.n_embd))
            for _ in range(self.config.n_layer)
        ]
        z = [
            torch.zeros(size=(1, self.config.feature_size, self.config.n_head))
            for _ in range(self.config.n_layer)
        ]
        return s, z

    def rnn_forward(self, x, s_rnn, z_rnn, i):
        # x, s, z are tensors
        # i is used for positional embedding
        hidden_states = self.wte(x) + self.wpe.weight.data[i].unsqueeze(0)
        # pass through GPT2Rnnblocks
        new_s_rnn, new_z_rnn = [], []
        for i, (GPT2Rnnblock, s, z) in enumerate(zip(self.h, s_rnn, z_rnn)):
            hidden_states, s, z = GPT2Rnnblock(
                hidden_states=hidden_states,
                attention_mask=None,
                s=s, z=z
            )
            new_s_rnn.append(s)
            new_z_rnn.append(z)
        output = self.ln_f(hidden_states)
        logits = self.lm_head(output)

        return logits, new_s_rnn, new_z_rnn

    # auto manage which forward method to use
    def forward(self, input_ids, attention_mask=None, labels=None, s_rnn=None, z_rnn=None, i=None):
        """auto manage this forward pass
        * for RNN mode: `input_ids, s_rnn, z_rnn, i` should be provided
        * for Transformer mode: `input_ids, attention_mask, labels` should be provided
        """
        if self.infer_ready_flag:
            assert s_rnn is not None, "In RNN mode need to provide `s` for each layer"
            assert z_rnn is not None, "In RNN mode need to provide `z` for each layer"
            assert i is not None, "In RNN mode need to provide step number `i`"
            # logits, new_s_rnn, new_z_rnn
            output = self.rnn_forward(input_ids, s_rnn, z_rnn, i)
        else:
            output = self.standard_forward(
                input_ids, attention_mask, labels)  # logits, (loss)
        return output


if __name__ == "__main__":
    # define config that will be used throughout this test
    tinyconf = GPT2Config(
        vocab_size=38,
        n_positions=32,
        n_ctx=32,
        n_embd=33,
        n_layer=2,
        n_head=11,
        feature_size=2
    )

    # test RNNAttention GPT2RnnBlock
    print("-" * 70)
    print("Testing RNNAttention")

    rnn = RNNAttention(tinyconf)
    output = rnn(torch.randn(1, 3, tinyconf.n_embd))
    assert len(output) == 1

    logits = output[0]
    assert tuple(logits.shape) == (1, 3, tinyconf.n_embd)
    rnn.infer_init()

    s, z = 0, 0
    for _ in range(tinyconf.n_ctx):
        out, s, z = rnn(x=torch.randn(1, 1, tinyconf.n_embd), s=s, z=z)
        assert tuple(out.shape) == (1, 1, tinyconf.n_embd)
        assert tuple(s.shape) == (1, tinyconf.feature_size, tinyconf.n_embd)
        assert tuple(z.shape) == (1, tinyconf.feature_size, tinyconf.n_head)
        if torch.isnan(out).any():
            raise ValueError("nan found, please check")
    print("... Passed!")

    # test T2R Model (Training Mode)
    print("-" * 70)
    print("Testing T2R Model (Training Mode)")

    t2r = GPT2T2R(tinyconf)
    output = t2r(
        input_ids=torch.randint(low=0, high=tinyconf.vocab_size, size=(1, 10))
    )
    assert len(output) == 1
    logits = output[0]
    assert tuple(logits.shape) == (1, 10, tinyconf.vocab_size)

    output = t2r(
        input_ids=torch.randint(low=0, high=tinyconf.vocab_size, size=(1, 10)),
        labels=torch.randint(low=0, high=tinyconf.vocab_size, size=(1, 10))
    )
    assert len(output) == 2
    loss, logits = output
    assert tuple(logits.shape) == (1, 10, tinyconf.vocab_size)
    print("... Passed!")

    # test T2R Model (Inference Mode)
    print("-" * 70)
    print("Testing T2R Model (Inference Mode)")

    t2r.infer_init()
    s_rnn = [0 for _ in range(tinyconf.n_layer)]
    z_rnn = [0 for _ in range(tinyconf.n_layer)]
    output = t2r(
        input_ids=torch.randint(low=0, high=tinyconf.vocab_size, size=(1, 1)),
        s_rnn=s_rnn, z_rnn=z_rnn, i=0
    )
    assert len(output) == 3
    logits, s_rnn, z_rnn = output
    assert tuple(logits.shape) == (1, 1, tinyconf.vocab_size)
    for s, z in zip(s_rnn, z_rnn):
        # s.shape == [B,k,E] & z.shape == [B,k,H]
        assert tuple(s.shape) == (1, tinyconf.feature_size, tinyconf.n_embd)
        assert tuple(z.shape) == (1, tinyconf.feature_size, tinyconf.n_head)

    s_rnn = [0 for _ in range(tinyconf.n_layer)]
    z_rnn = [0 for _ in range(tinyconf.n_layer)]
    for i in range(tinyconf.n_ctx):
        logits, s_rnn, z_rnn = t2r(
            input_ids=logits.argmax(-1),
            s_rnn=s_rnn, z_rnn=z_rnn, i=i
        )
        if torch.isnan(logits).any():
            raise ValueError("nan found, please check")
    print("... Passed!")

    # test Generation Mixin
    print("-" * 70)
    print("Testing Generation Mixin")

    input_ids = torch.randint(low=0, high=tinyconf.vocab_size, size=(1, 1))
    generated_seqs = t2r.generate(
        input_ids=input_ids,
        max_length=tinyconf.n_ctx,
        num_return_sequences=10
    )
    assert tuple(generated_seqs.shape) == (10, tinyconf.n_ctx)
    print("... Passed!")
