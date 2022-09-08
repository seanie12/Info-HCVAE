import argparse
import pickle

import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
from models import DiscreteVAE
from squad_utils import (InputFeatures, convert_examples_to_harv_features,
                         read_examples, read_squad_examples)


def return_mask_lengths(ids):
    mask = torch.sign(ids).float()
    lengths = torch.sum(mask, 1).long()
    return mask, lengths


def post_process(q_ids, start_positions, end_positions, c_ids, total_max_len=384):
    """
       concatenate question and context for BERT QA model:
       [CLS] Question [SEP] Context [SEP]
    """
    batch_size = q_ids.size(0)
    # exclude CLS token in c_ids
    c_ids = c_ids[:, 1:]
    start_positions = start_positions - 1
    end_positions = end_positions - 1

    _, q_lengths = return_mask_lengths(q_ids)
    _, c_lengths = return_mask_lengths(c_ids)

    all_input_ids = []
    all_seg_ids = []
    for i in range(batch_size):
        q_length = q_lengths[i]
        c_length = c_lengths[i]
        q = q_ids[i, :q_length]  # exclude pad tokens
        c = c_ids[i, :c_length]  # exclude pad tokens

        # input ids
        pads = torch.zeros((total_max_len - q_length - c_length), device=q_ids.device, dtype=torch.long)
        input_ids = torch.cat([q, c, pads], dim=0)
        all_input_ids.append(input_ids)

        # segment ids
        zeros = torch.zeros_like(q)
        ones = torch.ones_like(c)
        seg_ids = torch.cat([zeros, ones, pads], dim=0)
        all_seg_ids.append(seg_ids)

        start_positions[i] = start_positions[i] + q_length
        end_positions[i] = end_positions[i] + q_length

    all_input_ids = torch.stack(all_input_ids, dim=0)
    all_seg_ids = torch.stack(all_seg_ids, dim=0)
    all_input_mask = (all_input_ids != 0).byte()

    return all_input_ids, all_seg_ids, all_input_mask, start_positions, end_positions


def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    args.tokenizer = tokenizer

    device = torch.cuda.current_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    vae = DiscreteVAE(checkpoint["args"])
    vae.load_state_dict(checkpoint["state_dict"])
    vae.eval()
    vae = vae.to(device)
    
    if args.squad:
        examples = read_squad_examples(args.data_file, is_training=True, debug=args.debug)
        features = convert_examples_to_harv_features(examples,
                                                     tokenizer=tokenizer,
                                                     max_seq_length=args.max_c_len,
                                                     max_query_length=args.max_q_len,
                                                     doc_stride=128,
                                                     is_training=True)
    else:
        examples = read_examples(args.data_file, is_training=True, debug=args.debug)
        features = convert_examples_to_harv_features(examples,
                                                     tokenizer=tokenizer,
                                                     max_seq_length=args.max_c_len,
                                                     max_query_length=args.max_q_len,
                                                     doc_stride=128,
                                                     is_training=True)

    features = features[:int(len(features) * args.ratio)]
    all_c_ids = torch.tensor([f.c_ids for f in features], dtype=torch.long)
    data = TensorDataset(all_c_ids)
    data_loader = DataLoader(data, shuffle=False, batch_size=args.batch_size)

    
    
    new_features = []

    for batch in tqdm(data_loader, total=len(data_loader)):
        c_ids = batch[0]
        _, c_len = return_mask_lengths(c_ids)
        max_c_len = torch.max(c_len)
        c_ids = c_ids[:, :max_c_len].to(device)
        
        # sample latent variable K times
        for _ in range(args.k):
            with torch.no_grad():
                _, _, zq, _, za = vae.prior_encoder(c_ids)
                batch_q_ids, batch_start, batch_end = vae.generate(zq, za, c_ids)

                all_input_ids, all_seg_ids, \
                all_input_mask, all_start, all_end = post_process(batch_q_ids, batch_start, batch_end, c_ids)

            for i in range(c_ids.size(0)):
                new_features.append(
                    InputFeatures(
                        unique_id=None,
                        example_index=None,
                        doc_span_index=None,
                        tokens=None,
                        token_to_orig_map=None,
                        token_is_max_context=None,
                        input_ids=all_input_ids[i].cpu().tolist(),
                        input_mask=all_input_mask[i].cpu().tolist(),
                        c_ids=None,
                        context_tokens=None,
                        q_ids=None,
                        q_tokens=None,
                        answer_text=None,
                        tag_ids=None,
                        segment_ids=all_seg_ids[i].cpu().tolist(),
                        noq_start_position=None,
                        noq_end_position=None,
                        start_position=all_start[i].cpu().tolist(),
                        end_position=all_end[i].cpu().tolist(),
                        is_impossible=None))

    dir_name = os.path.dirname(args.output_file)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(args.output_file, "wb") as f:
        pickle.dump(new_features, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--squad', dest='squad', action='store_true', help="whether to generate QA from SQuAD context")

    parser.add_argument("--seed", default=1004, type=int)
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str)
    parser.add_argument("--max_c_len", default=384 - 64, type=int, help="max context length")
    parser.add_argument("--max_q_len", default=0, type=int, help="max query length")

    parser.add_argument("--batch_size", default=64, type=int, help="batch_size")
    parser.add_argument("--data_file", default="../data/squad/train-v1.1.json", type=str)
    parser.add_argument("--checkpoint", default="../save/vae-checkpoint/best_f1_model.pt", type=str, help="checkpoint for vae model")
    parser.add_argument("--output_file", default="../data/synthetic_data/1.0_squad_10x_features.pkl", type=str)

    parser.add_argument("--ratio", default=1.0, type=float)
    parser.add_argument("--k", default=10, type=int, help="the number of QA pairs for each paragraph")

    args = parser.parse_args()
    main(args)
