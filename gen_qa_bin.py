import argparse
import math
import h5py

import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
from infohcvae.model.qag_vae import DiscreteVAE
from infohcvae.squad_utils import (convert_examples_to_harv_features,
                                   read_examples, read_squad_examples)


def return_mask_lengths(ids):
    mask = torch.sign(ids).float()
    lengths = torch.sum(mask, 1).long()
    return mask, lengths


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-20, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor

    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()  # ~Gumbel(0,1), shape=(batch, nza, nzadim)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau), shape=(batch, nza, nzadim)
    y_soft = gumbels.softmax(dim) # shape=(batch, nza, nzadim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1] # shape = (batch, nza, 1)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0) # sampling one-hot categorical variables
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Re-parametrization trick.
        ret = y_soft
    return ret


def post_process(q_ids, start_positions, end_positions, c_ids, total_max_len=512):
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
    tokenizer = AutoTokenizer.from_pretrained(args.huggingface_model)
    args.tokenizer = tokenizer

    device = torch.cuda.current_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    vae = DiscreteVAE(checkpoint["args"])
    vae.load_state_dict(checkpoint["vae_state_dict"])
    vae.eval()
    vae = vae.to(device)

    data_loader = None
    if not args.load_saved_dataloader:
        # Add shuffling functionality if wanting to use a small percentage of data correctly
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
        all_c_ids = torch.tensor([f.c_ids for f in features][:int(args.data_ratio*len(features))], dtype=torch.long)
        data = TensorDataset(all_c_ids)
        data_loader = DataLoader(data, shuffle=False, batch_size=args.batch_size)
        print("Dataset length = " + str(len(data_loader.dataset)))
        torch.save(data_loader, os.path.join(args.dataloader_dir, "gen_loader.pt"))
    else:
        data_loader = torch.load(os.path.join(args.dataloader_dir, "gen_loader.pt"))
        print("Dataset length = " + str(len(data_loader.dataset)))

    with h5py.File(args.output_file, "a") as fdata:
        input_ids_set = fdata.create_dataset("qas/input_ids", (len(data_loader.dataset)*10, args.total_max_len),
                                                            chunks=(100, args.total_max_len))
        input_masks_set = fdata.create_dataset("qas/input_masks", (len(data_loader.dataset)*10, args.total_max_len),
                                                            chunks=(100, args.total_max_len))
        segment_ids_set = fdata.create_dataset("qas/segment_ids", (len(data_loader.dataset)*10, args.total_max_len),
                                                            chunks=(100, args.total_max_len))
        start_positions_set = fdata.create_dataset("qas/start_positions", (len(data_loader.dataset)*10,), chunks=(1000,))
        end_positions_set = fdata.create_dataset("qas/end_positions", (len(data_loader.dataset)*10,), chunks=(1000,))

        # new_features = []
        qa_text = None
        if args.out_qa_json is not None and args.output_text:
            qa_text = dict({"data": []})

        num_steps_to_run = math.ceil(args.percent_of_runs * len(data_loader))
        print("Num steps to run: {:d}".format(num_steps_to_run))
        step = 0
        qa_idx = 0
        for batch in tqdm(data_loader, total=len(data_loader)):
            step += 1
            if step < args.resume_steps:
                continue

            if num_steps_to_run == 0:
                break

            num_steps_to_run = num_steps_to_run - 1

            c_ids = batch[0]
            _, c_len = return_mask_lengths(c_ids)
            max_c_len = torch.max(c_len)
            c_ids = c_ids[:, :max_c_len].to(device)

            c_texts = [args.tokenizer.decode(c_ids[idx]) for idx in range(c_ids.size(0))]

            # sample latent variable K times
            for idx in range(args.k):
                zq, za = vae.prior_encoder(c_ids)

                with torch.no_grad():
                    batch_q_ids, batch_start, batch_end = vae.generate(zq, za, c_ids)
                    # batch_q_ids, batch_start, batch_end = vae.generate(c_ids)

                    if args.out_qa_json is not None and args.output_text: # out QA text to json
                        for idx in range(batch_q_ids.size(0)):
                            q_ids, start_pos, end_pos = batch_q_ids[idx], batch_start[idx], batch_end[idx]
                            q_text = args.tokenizer.decode(q_ids)
                            ans_text = args.tokenizer.decode(c_ids[idx, start_pos:end_pos])
                            qa_text["data"].append({"context": c_texts[idx], "question": q_text, "answer": ans_text})

                    all_input_ids, all_seg_ids, \
                    all_input_mask, all_start, all_end = post_process(batch_q_ids, batch_start, batch_end, c_ids, total_max_len=args.total_max_len)

                for i in range(c_ids.size(0)):
                    input_ids_set[qa_idx, :] = all_input_ids[i].cpu()
                    input_masks_set[qa_idx, :] = all_input_mask[i].cpu()
                    segment_ids_set[qa_idx, :] = all_seg_ids[i].cpu()
                    start_positions_set[qa_idx] = all_start[i].cpu()
                    end_positions_set[qa_idx] = all_end[i].cpu()
                    qa_idx += 1

    ## For outputting text
    if args.output_text:
        import json
        dir_name = os.path.dirname(args.out_qa_json)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(args.out_qa_json, "wt") as f:
            json.dump(qa_text, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--squad', dest='squad', action='store_true', help="whether to generate QA from SQuAD context")
    parser.add_argument("--load_saved_dataloader", dest="load_saved_dataloader", action="store_true")
    parser.add_argument("--output_text", dest="output_text", action="store_true")

    parser.add_argument("--seed", default=1004, type=int)
    parser.add_argument("--huggingface_model", default='bert-base-uncased', type=str)
    parser.add_argument("--resume_steps", default=1, type=int, help="step to resume")
    parser.add_argument("--percent_of_runs", default=1.0, type=float, help="how many percent of steps to run at one execution")
    parser.add_argument("--vietnamese", default=False, type=bool)
    parser.add_argument("--max_c_len", default=384 - 64, type=int, help="max context length")
    parser.add_argument("--total_max_len", default=384, type=int, help="total max length")
    parser.add_argument("--max_q_len", default=0, type=int, help="max query length")

    parser.add_argument("--batch_size", default=64, type=int, help="batch_size")
    parser.add_argument("--data_file", default="../data/squad/train-v1.1.json", type=str)
    parser.add_argument("--checkpoint", default="../save/vae-checkpoint/best_f1_model.pt", type=str, help="checkpoint for vae model")
    parser.add_argument("--output_file", default="../data/1.0_squad_10x_features.h5", type=str)
    parser.add_argument("--out_qa_json", default="../data/generated_qas.json", type=str)
    parser.add_argument("--dataloader_dir", default="../save/dataloader", type=str)

    parser.add_argument("--data_ratio", default=1.0, type=float, help="how many percentage of the number of paragraphs are considered for generation")
    parser.add_argument("--ratio", default=1.0, type=float)
    parser.add_argument("--k", default=10, type=int, help="the number of QA pairs for each paragraph")

    args = parser.parse_args()

    args.load_saved_dataloader = True if args.load_saved_dataloader == "True" else False
    args.output_text = True if args.output_text == "True" else False

    # set dataloader dir
    if not args.load_saved_dataloader:
        dataloader_dir = args.dataloader_dir
        os.makedirs(dataloader_dir, exist_ok=True)
        args.dataloader_dir = os.path.abspath(dataloader_dir)

    main(args)