import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm, trange
from transformers import BertTokenizer

from eval import eval_vae
from trainer import VAETrainer
from utils import batch_to_device, get_harv_data_loader, get_squad_data_loader


def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.huggingface_model)
    train_loader = None
    eval_data = None
    
    if args.load_saved_dataloader:
        train_loader = torch.load(os.path.join(args.dataloader_dir, "train_loader.pt"))
        eval_data = torch.load(os.path.join(args.dataloader_dir, "eval_loader.pt"))
    else:
        train_loader, _, _ = get_squad_data_loader(tokenizer, args.train_dir,
                                         shuffle=True, is_train_set=True, args=args)
        eval_data = get_squad_data_loader(tokenizer, args.dev_dir,
                                          shuffle=False, is_train_set=False, args=args)
        torch.save(train_loader, os.path.join(args.dataloader_dir, "train_loader.pt"))
        torch.save(eval_data, os.path.join(args.dataloader_dir, "eval_loader.pt"))

    args.device = torch.cuda.current_device()

    trainer = VAETrainer(args)
    if args.checkpoint_file is not None:
        trainer.load_model_state_dict(args.checkpoint_file)

    # loss_log1 = tqdm(total=0, bar_format='{desc}', position=2)
    # loss_log2 = tqdm(total=0, bar_format='{desc}', position=3)
    # eval_log = tqdm(total=0, bar_format='{desc}', position=5)
    # best_eval_log = tqdm(total=0, bar_format='{desc}', position=6)

    print("MODEL DIR: " + args.model_dir)

    num_samples_limit = 1000000000
    if args.is_test_run:
        num_samples_limit = 2000

    best_bleu, best_em, best_f1 = args.prev_best_bleu, 0.0, args.prev_best_f1
    for epoch in trange(int(args.epochs), desc="Epoch", position=0):
        if epoch+1 < args.resume_epochs:
            continue

        trainer.reset_cnt_steps()
        cnt_samples = 0
        for batch in tqdm(train_loader, leave=False, position=1):
            c_ids, q_ids, a_ids, start_positions, end_positions \
                = batch_to_device(batch, args.device)
            trainer.train(c_ids, q_ids, a_ids, start_positions, end_positions)

            # str1 = 'Q REC : {:.6f} A REC : {:.6f}'
            # str2 = 'ZQ KL : {:.6f} ZA KL : {:.6f} ZQ MMD : {:.6f} ZA MMD : {:.6f} INFO : {:.6f}'
            # str1 = str1.format(float(trainer.loss_q_rec), float(trainer.loss_a_rec))
            # str2 = str2.format(float(trainer.loss_zq_kl), float(trainer.loss_za_kl), float(trainer.loss_zq_mmd), float(trainer.loss_za_mmd), float(trainer.loss_info))
            # loss_log1.set_description_str(str1)
            # loss_log2.set_description_str(str2)

            cnt_samples += c_ids.shape[0] # add batch dimension to get number of samples
            if cnt_samples >= num_samples_limit:
                # stop training if over the num of training samples limit
                break

        trainer.print_log(log_type="epoch")

        if (epoch + 1) % args.eval_freq == 0:
            metric_dict, bleu, _ = eval_vae(epoch, args, trainer, eval_data)
            f1 = metric_dict["f1"]
            em = metric_dict["exact_match"]
            bleu = bleu * 100
            log_str = '{}-th Epochs BLEU : {:02.2f} EM : {:02.2f} F1 : {:02.2f}'
            log_str = log_str.format(epoch, bleu, em, f1)
            print(log_str)
            if em > best_em:
                best_em = em
            if f1 > best_f1:
                best_f1 = f1
                trainer.save(os.path.join(args.best_model_dir, "best_f1_model.pt"))
            if bleu > best_bleu:
                best_bleu = bleu
                trainer.save(os.path.join(args.best_model_dir, "best_bleu_model.pt"))

            log_str = 'BEST BLEU : {:02.2f} EM : {:02.2f} F1 : {:02.2f}'
            log_str = log_str.format(best_bleu, best_em, best_f1)
            print(log_str)

            with open(os.path.join(args.model_dir, "metrics.json"), "wt") as f:
                import json
                json.dump({"best_bleu": best_bleu, "best_em": best_em, "best_f1": best_f1}, f, indent=4)

        if (epoch + 1) % args.save_freq == 0:
            trainer.save(os.path.join(args.save_by_epoch_dir, "model-epoch-{:02d}.pt".format(epoch + 1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1004, type=int)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--train_dir', default='../data/squad/train-v1.1.json')
    parser.add_argument('--dev_dir', default='../data/squad/my_dev.json')
    parser.add_argument("--train_percentage", default=100, type=int, help="training data percentage of questions to use")
    
    parser.add_argument("--max_c_len", default=384, type=int, help="max context length")
    parser.add_argument("--max_q_len", default=64, type=int, help="max query length")
    parser.add_argument("--load_saved_dataloader", default="False", type=str)
    parser.add_argument("--use_mine", default="False", type=str)

    parser.add_argument("--model_dir", default="../save/vae-checkpoint", type=str)
    parser.add_argument("--dataloader_dir", default="../save/dataloader", type=str)
    parser.add_argument("--checkpoint_file", default=None, type=str, help="Path to the .pt file, None if checkpoint should not be loaded")
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--resume_epochs", default=1, type=int)
    parser.add_argument("--is_test_run", default="False", type=str)
    parser.add_argument("--prev_best_bleu", default=0.0, type=float)
    parser.add_argument("--prev_best_f1", default=0.0, type=float)
    parser.add_argument("--save_freq", default=2, type=int, help="Model saving should be executed after how many epochs?")
    parser.add_argument("--eval_freq", default=10, type=int, help="Model validation should be executed after how many epochs?")
    parser.add_argument("--lr", default=1e-3, type=float, help="lr")
    parser.add_argument("--lr_infomax", default=1e-5, type=float, help="lr for infomax net")
    parser.add_argument("--batch_size", default=64, type=int, help="batch_size")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay")
    parser.add_argument("--clip", default=5.0, type=float, help="max grad norm")

    parser.add_argument("--huggingface_model", default='bert-base-uncased', type=str)
    parser.add_argument('--enc_nhidden', type=int, default=300)
    parser.add_argument('--enc_nlayers', type=int, default=1)
    parser.add_argument('--enc_dropout', type=float, default=0.2)
    parser.add_argument('--dec_a_nhidden', type=int, default=300)
    parser.add_argument('--dec_a_nlayers', type=int, default=1)
    parser.add_argument('--dec_a_dropout', type=float, default=0.2)
    parser.add_argument('--dec_q_nhidden', type=int, default=900)
    parser.add_argument('--dec_q_nlayers', type=int, default=2)
    parser.add_argument('--dec_q_dropout', type=float, default=0.3)
    parser.add_argument('--nzqdim', type=int, default=50)
    parser.add_argument('--nza', type=int, default=20)
    parser.add_argument('--nzadim', type=int, default=10)
    parser.add_argument('--w_ans', type=float, default=1.0)
    parser.add_argument('--w_bce', type=float, default=1.0)
    parser.add_argument('--alpha_kl', type=float, default=0.0)
    # parser.add_argument('--lambda_mmd', type=float, default=1.0)
    parser.add_argument('--lambda_z_info', type=float, default=10.0)
    parser.add_argument('--lambda_qa_info', type=float, default=10.0)

    args = parser.parse_args()

    if args.debug:
        args.model_dir = "./dummy"

    # Determine boolean args
    args.load_saved_dataloader = True if args.load_saved_dataloader == "True" else False
    args.use_mine = True if args.use_mine == "True" else False
    args.is_test_run = True if args.is_test_run == "True" else False

    # set model dir
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    args.model_dir = os.path.abspath(model_dir)

    save_by_epoch_dir = os.path.join(args.model_dir, "per_epoch")
    best_model_dir = os.path.join(args.model_dir, "best_models")
    os.makedirs(save_by_epoch_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    args.save_by_epoch_dir = save_by_epoch_dir
    args.best_model_dir = best_model_dir

    # set dataloader dir
    if not args.load_saved_dataloader:
        dataloader_dir = args.dataloader_dir
        os.makedirs(dataloader_dir, exist_ok=True)
        args.dataloader_dir = os.path.abspath(dataloader_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
