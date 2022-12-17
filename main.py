import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm, trange
from transformers import BertTokenizer

from infohcvae.qagen_eval import eval_vae
from infohcvae.trainer import VAETrainer
from infohcvae.utils import batch_to_device, get_squad_data_loader, generate_testing_dataset_for_model_choosing


def evaluate_model(epoch, args, trainer, eval_data, best_bleu, best_em, best_f1):
    posterior_metrics, prior_metrics, bleu = eval_vae(args, trainer, eval_data)
    posterior_f1 = posterior_metrics["f1"]
    posterior_em = posterior_metrics["exact_match"]
    prior_f1 = prior_metrics["f1"]
    prior_em = prior_metrics["exact_match"]
    bleu = bleu * 100

    if posterior_em > best_em:
        best_em = posterior_em
    if posterior_f1 > best_f1:
        best_f1 = posterior_f1
        trainer.save(args.best_model_dir, save_mode="best_f1")
    if bleu > best_bleu:
        best_bleu = bleu
        trainer.save(args.best_model_dir, save_mode="best_bleu")

    with open(os.path.join(args.model_dir, "metrics.json"), "wt") as f:
        import json
        json.dump({ "latest_bleu": bleu, "latest_pos_em": posterior_em, "latest_pos_f1": posterior_f1,
                    "latest_pri_em": prior_em, "latest_pri_f1": prior_f1,
                    "best_bleu": best_bleu, "best_em": best_em, "best_f1": best_f1 }, f, indent=4)

    log_str = "{}-th Epochs BLEU : {:02.2f} POS_EM : {:02.2f} POS_F1 : {:02.2f} " + \
              "PRI_EM : {:02.2f} PRI_F1 : {:02.2f}"
    log_str = log_str.format(epoch, bleu, posterior_em, posterior_f1, prior_em, prior_f1)
    print(log_str)

    log_str = "BEST BLEU : {:02.2f} EM : {:02.2f} F1 : {:02.2f}"
    log_str = log_str.format(best_bleu, best_em, best_f1)
    print(log_str)


def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.huggingface_model)

    train_data, eval_data = None, None
    if args.load_saved_dataloader:
        train_data = torch.load(os.path.join(args.dataloader_dir, "train_data.pt"))
        eval_data = torch.load(os.path.join(args.dataloader_dir, "eval_data.pt"))
    else:
        train_data = get_squad_data_loader(tokenizer, args.train_dir,
                                         shuffle=True, is_train_set=True, args=args)
        eval_data = get_squad_data_loader(tokenizer, args.dev_dir,
                                          shuffle=False, is_train_set=False, args=args)
        torch.save(train_data, os.path.join(args.dataloader_dir, "train_data.pt"))
        torch.save(eval_data, os.path.join(args.dataloader_dir, "eval_data.pt"))

    args.device = torch.cuda.current_device()

    trainer = VAETrainer(args)
    if args.checkpoint_file is not None:
        trainer.load_model_state_dict(args.checkpoint_file)

    print("MODEL DIR: " + args.model_dir)

    if args.is_test_run:
        test_train_data, test_eval_data = generate_testing_dataset_for_model_choosing(train_data)
        train_data = test_train_data
        eval_data = test_eval_data

    train_loader, _, _ = train_data
    current_lr = args.lr
    best_bleu, best_em, best_f1 = 0.0, 0.0, 0.0
    first_run = True

    if os.path.exists(os.path.join(args.model_dir, "metrics.json")):
        with open(os.path.join(args.model_dir, "metrics.json")) as f:
            import json
            metrics = json.load(f)
        best_bleu = metrics["best_bleu"]
        best_em = metrics["best_em"]
        best_f1 = metrics["best_f1"]

    if args.resume_epochs > 1 and args.eval_before_resume:
        evaluate_model(args.resume_epochs - 1, args, trainer, eval_data, best_bleu, best_em, best_f1)

    for epoch in trange(int(args.epochs), desc="Epoch", position=0):
        if epoch+1 < args.resume_epochs:
            continue

        if not args.is_test_run and args.optimizer == "manual":
            if epoch + 1 >= 31:
                current_lr = current_lr / 10
                trainer.change_optimizer(args, optimizer="sgd", lr=current_lr, weight_decay=args.weight_decay)
            elif epoch + 1 >= 21:
                # reduce by 10 at epoch > 20 = 0.00001 = 1e-5
                current_lr = current_lr / 2
                trainer.change_optimizer(args, optimizer="sgd", lr=current_lr, weight_decay=args.weight_decay)
            elif epoch+1 >= 15:
                # reduce the LR by 5 times at epoch 15 = 0.0001 = 1e-4
                current_lr = current_lr / 2
                trainer.change_optimizer(args, optimizer="sgd", lr=current_lr, weight_decay=args.weight_decay)
            elif epoch+1 >= 11:
                # change optimizer to sgd
                trainer.change_optimizer(args, optimizer="sgd", lr=current_lr, weight_decay=args.weight_decay)
            elif epoch+1 >= 5:
                # boost grad descent by reset Adam with LR / 2 = 0.0005 = 5e-4
                current_lr = current_lr / 2
                trainer.change_optimizer(args, optimizer="adam", lr=current_lr, weight_decay=args.weight_decay)

        trainer.reset_cnt_steps()
        for batch in tqdm(train_loader, leave=False, position=1):
            c_ids, q_ids, a_ids, start_positions, end_positions \
                = batch_to_device(batch, args.device)
            trainer.train(c_ids, q_ids, a_ids, start_positions, end_positions)

            if epoch == 0 and first_run: # first iteration
                trainer.print_log() # get first run loss to verify correctness
                first_run = False

        trainer.print_log(log_type="epoch", epoch=epoch+1)

        # Save before evaluation to make sure that if evaluating has errors, model is still saved beforehand
        if (epoch + 1) % args.save_freq == 0:
            trainer.save(args.save_by_epoch_dir, epoch=epoch+1, save_freq=args.save_freq)

        if (epoch + 1) % args.eval_freq == 0:
            evaluate_model(epoch, args, trainer, eval_data, best_bleu, best_em, best_f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1004, type=int)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--train_dir', default='../data/squad/train-v1.1.json')
    parser.add_argument('--dev_dir', default='../data/squad/my_dev.json')
    parser.add_argument("--train_percentage", default=100, type=int, help="training data percentage of questions to use")

    parser.add_argument("--max_c_len", default=384, type=int, help="max context length")
    parser.add_argument("--max_q_len", default=64, type=int, help="max query length")
    parser.add_argument("--load_saved_dataloader", dest="load_saved_dataloader", action="store_true")

    parser.add_argument("--model_dir", default="../save/vae-checkpoint", type=str)
    parser.add_argument("--dataloader_dir", default="../save/dataloader", type=str)
    parser.add_argument("--checkpoint_file", default=None, type=str, help="Path to the .pt file, None if checkpoint should not be loaded")
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--resume_epochs", default=1, type=int)
    parser.add_argument("--is_test_run", dest="is_test_run", action="store_true")
    parser.add_argument("--log_loss_info", dest="log_loss_info", action="store_true")
    parser.add_argument("--eval_before_resume", dest="eval_before_resume", action="store_true", default=False)
    parser.add_argument("--save_freq", default=5, type=int, help="Model saving should be executed after how many epochs?")
    parser.add_argument("--eval_freq", default=5, type=int, help="Model validation should be executed after how many epochs?")
    parser.add_argument("--lr", default=1e-3, type=float, help="lr")
    parser.add_argument("--optimizer", default="manual", choices=["sgd", "adam", "swats", "manual"], type=str, \
        help="optimizer to use, [\"adam\", \"sgd\", \"swats\", \"manual\"] are supported")
    parser.add_argument("--batch_size", default=64, type=int, help="batch_size")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay")
    parser.add_argument("--clip", default=5.0, type=float, help="max grad norm")

    parser.add_argument("--huggingface_model", default='bert-base-uncased', type=str)
    parser.add_argument('--enc_nhidden', type=int, default=300)
    parser.add_argument('--enc_nlayers', type=int, default=1)
    parser.add_argument('--enc_dropout', type=float, default=0.3)
    parser.add_argument('--dec_a_nhidden', type=int, default=300)
    parser.add_argument('--dec_a_nlayers', type=int, default=1)
    parser.add_argument('--dec_a_dropout', type=float, default=0.3)
    parser.add_argument('--dec_q_nhidden', type=int, default=900)
    parser.add_argument('--dec_q_nlayers', type=int, default=2)
    parser.add_argument('--dec_q_dropout', type=float, default=0.3)
    parser.add_argument('--nzqdim', type=int, default=64)
    parser.add_argument('--nza', type=int, default=32)
    parser.add_argument('--nzadim', type=int, default=16)
    parser.add_argument('--w_bce', type=float, default=1.5)
    parser.add_argument('--alpha_kl', type=float, default=0.9)
    parser.add_argument('--lambda_mmd_q', type=float, default=1.6)
    parser.add_argument('--lambda_mmd_a', type=float, default=1.6)
    parser.add_argument('--lambda_qa_info', type=float, default=1.5)

    args = parser.parse_args()

    if args.debug:
        args.model_dir = "./dummy"

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

    if args.log_loss_info:
        open("loss_info.log", "w") # empty loss log file if existed

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)