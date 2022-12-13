import os
import torch
from model.qag_vae import DiscreteVAE
import torch_optimizer as additional_optim
import torch.optim as optim


class VAETrainer(object):
    def __init__(self, args):
        self.args = args
        self.clip = args.clip
        self.device = args.device
        self.log_loss_info = args.log_loss_info

        self.vae = DiscreteVAE(args).to(self.device)
        self.params = filter(lambda p: p.requires_grad, self.vae.parameters())
        # self.params = self.vae.get_vae_params(lr=args.lr) + (self.vae.get_infomax_params(lr=args.lr/100) if args.lambda_z_info > 0 else [])
        if args.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.params, lr=args.lr, momentum=0.9, nesterov=False, weight_decay=args.weight_decay)
        elif args.optimizer == "adam" or args.optimizer == "manual":
            self.optimizer = optim.Adam(
                self.params, lr=args.lr, weight_decay=args.weight_decay)
        else:
            self.optimizer = additional_optim.SWATS(
                self.params, lr=args.lr, nesterov=False, weight_decay=args.weight_decay)

        self.losses = dict()
        self.cnt_steps = 0

    def adjust_infomax_weight(self, infomax_loss):
        if infomax_loss > 1.0:
            self.vae.reduce_infomax_weight_by_10()

    def train(self, c_ids, q_ids, a_ids, start_positions, end_positions):
        self.vae.train()

        # Forward
        return_dict = self.vae(c_ids, q_ids, a_ids,
                               start_positions, end_positions)

        # Backward
        self.optimizer.zero_grad()
        return_dict["total_loss"].backward()
        # Step
        self.optimizer.step()

        for key, value in return_dict.items():
            if key in self.losses:
                self.losses[key] += value.item()
            else:
                self.losses[key] = value.item()

        self.cnt_steps += 1
        if self.cnt_steps % 50 == 0:
            self.print_log()

        # self.adjust_infomax_weight(return_dict["loss_z_info"].item())

    def change_optimizer(self, args, optimizer="adam", lr=1e-4, weight_decay=0.0):
        assert optimizer in ["sgd", "adam", "swats"]
        self.params = filter(lambda p: p.requires_grad, self.vae.parameters())
        # self.params = self.vae.get_vae_params(lr=lr) + (self.vae.get_infomax_params(lr=lr/100) if args.lambda_z_info > 0 else [])
        if optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
        elif optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = additional_optim.SWATS(
                self.params, lr=args.lr, nesterov=False, weight_decay=args.weight_decay)

    def print_log(self, log_type="step", epoch=None):
        """
            log_type: "epoch" or "step"
            epoch: enter a number
        """
        assert log_type in ["step", "epoch"]
        assert log_type == "step" or (
            log_type == "epoch" and epoch is not None)

        log_str = ""
        if log_type == "step":
            log_str = "\nStep={:d} - ".format(self.cnt_steps)
        else:
            log_str = "\nEpoch={:d} - ".format(epoch)

        for key, value in self.losses.items():
            log_str += "{:s}={:.4f}; ".format(key, value / self.cnt_steps)

        print(log_str + "\n")

        if self.log_loss_info:  # log to file for monitoring
            with open("loss_info.log", "a") as f:  # open loss log in the current working dir
                f.write(("\n\n" if log_type == "epoch" else "") +
                        log_str + "\n" +
                        ("\n" if log_type == "epoch" else ""))

    def _reset_loss_values(self):
        for key in self.losses.keys():
            self.losses[key] = 0

    def reset_cnt_steps(self):
        self.cnt_steps = 0
        self._reset_loss_values()

    def generate_posterior(self, c_ids, q_ids, a_ids):
        with torch.no_grad():
            zq, za = self.vae.posterior_encoder(c_ids, q_ids, a_ids)
            q_ids, start_positions, end_positions = self.vae.generate(
                zq, za, c_ids)
        return q_ids, start_positions, end_positions, zq

    def generate_answer_logits(self, c_ids, q_ids, a_ids):
        with torch.no_grad():
            zq, za = self.vae.posterior_encoder(c_ids, q_ids, a_ids)
            start_logits, end_logits = self.vae.return_answer_logits(
                zq, za, c_ids)
        return start_logits, end_logits
    
    def generate_prior_answer_logits(self, c_ids, q_ids, a_ids):
        """Generate prior answer logits based on posterior_zq"""
        with torch.no_grad():
            zq, _ = self.vae.posterior_encoder(c_ids, q_ids, a_ids)
            _, za = self.vae.prior_encoder(c_ids, zq=zq)
            start_logits, end_logits = self.vae.return_answer_logits(
                zq, za, c_ids)
        return start_logits, end_logits

    def generate_prior(self, c_ids, posterior_zq=None):
        with torch.no_grad():
            zq, za = self.vae.prior_encoder(c_ids, zq=posterior_zq)
            q_ids, start_positions, end_positions = self.vae.generate(
                zq, za, c_ids)
        return q_ids, start_positions, end_positions, zq

    def save(self, save_path, save_mode="epoch", epoch=None, save_freq=None, max_models_to_keep=4):
        assert save_mode in ["best_f1", "best_bleu"] or \
            (save_mode == "epoch" and epoch is not None and save_freq is not None \
                and epoch % save_freq == 0)

        if save_mode == "epoch":
            filename = os.path.join(save_path, "model-epoch-{:02d}.pt".format(epoch))
        elif save_mode == "best_f1":
            filename = os.path.join(save_path, "best_f1_model.pt")
        else: # best_bleu
            filename = os.path.join(save_path, "best_bleu_model.pt")

        params = {
            "vae_state_dict": self.vae.state_dict(),
            "args": self.args
        }
        torch.save(params, filename)

        if save_mode == "epoch": # max model windows only applies to save_mode="epoch"
            if epoch - max_models_to_keep*save_freq >= 1:
                remove_filename = os.path.join(
                    save_path, "model-epoch-{:02d}.pt".format(epoch - max_models_to_keep))
                if os.path.exists(remove_filename):
                    os.remove(remove_filename)

    def load_model_state_dict(self, filename):
        params = torch.load(filename)
        self.vae.load_state_dict(params["vae_state_dict"])

    def set_eval_mode(self, enable=True):
        if enable:
            self.vae.eval()
