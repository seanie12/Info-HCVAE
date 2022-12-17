import argparse
import json
import os

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer

from infohcvae.model.qag_vae import DiscreteVAE


class CustomDatset(Dataset):
    def __init__(self, tokenizer, input_file, max_length=512):
        self.lines = open(input_file, "r").readlines()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_total = len(self.lines)

    def __getitem__(self, idx):
        context = self.lines[idx].strip()
        tokens = self.tokenizer.tokenize(context)[:self.max_length]
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding up to the maximum length
        while len(ids) < self.max_length:
            ids.append(0)
        ids = torch.tensor(ids, dtype=torch.long)
        
        return ids

    def __len__(self):
        return self.num_total


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.huggingface_model)
    data = CustomDatset(tokenizer, args.data_file, args.max_length)
    data_loader = DataLoader(data, shuffle=False, batch_size=args.batch_size)

    device = torch.cuda.current_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    vae = DiscreteVAE(checkpoint["args"])
    vae.load_state_dict(checkpoint["vae_state_dict"])
    vae.eval()
    vae = vae.to(device)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_file = os.path.join(args.output_dir, "synthetic_qa.jsonl")
    
    fw = open(output_file, "w")
    for batch in tqdm(data_loader, total=len(data_loader)):
        c_ids = batch
        c_len = torch.sum(torch.sign(c_ids),1 )
        max_c_len = torch.max(c_len)
        c_ids = c_ids[:, :max_c_len].to(device)

        # sample latent variable K times
        for _ in range(args.k):
            with torch.no_grad():
                zq, za = vae.prior_encoder(c_ids)
                batch_q_ids, batch_start, batch_end = vae.generate(
                    zq, za, c_ids)

            for i in range(c_ids.size(0)):
                _c_ids = c_ids[i].cpu().tolist()
                q_ids = batch_q_ids[i].cpu().tolist()
                start_pos = batch_start[i].item()
                end_pos = batch_end[i].item()
                
                a_ids = _c_ids[start_pos: end_pos+1]
                c_text = tokenizer.decode(_c_ids, replace_special_tokens=True)
                q_text = tokenizer.decode(q_ids, replace_speical_tokens=True)
                a_text = tokenizer.decode(a_ids, replace_special_tokens=True)
                json_dict = {
                    "context":c_text,
                    "question": q_text,
                    "answer": a_text
                }
                fw.write(json.dumps(json_dict) + "\n")
                fw.flush()

    fw.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1004, type=int)
    parser.add_argument("--huggingface_model", default='bert-base-uncased', type=str)
    parser.add_argument("--max_length", default=384,
                        type=int, help="max context length")
    
    parser.add_argument("--batch_size", default=64,
                        type=int, help="batch_size")
    parser.add_argument("--data_file", type=str,
                        required=True, help="text file of paragraphs")
    parser.add_argument("--checkpoint", default="../save/vae-checkpoint/best_f1_model.pt",
                        type=str, help="checkpoint for vae model")
    parser.add_argument("--output_dir", default="../data/synthetic_data/", type=str)

    parser.add_argument("--ratio", default=1.0, type=float)
    parser.add_argument("--k", default=1, type=int,
                        help="the number of QA pairs for each paragraph")

    args = parser.parse_args()
    main(args)
