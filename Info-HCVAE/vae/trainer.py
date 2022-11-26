import torch
import torch.nn as nn
from model.qag_vae import DiscreteVAE

class VAETrainer(object):
    def __init__(self, args):
        self.args = args
        self.clip = args.clip
        self.device = args.device

        self.vae = DiscreteVAE(args).to(self.device)
        params = filter(lambda p: p.requires_grad, self.parameters())
        # params = self.vae.get_vae_params(lr=args.lr) + (self.vae.get_infomax_params(lr=args.lr_infomax) if args.use_infomax_net else [])
        if args.use_sgd:
            self.optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

        self.losses = {
            "total_loss": 0,
            "loss_q_rec": 0,
            "loss_a_rec": 0,
            "loss_kl": 0,
            "loss_zq_kl": 0,
            "loss_za_kl": 0,
            "loss_mmd": 0,
            "loss_zq_mmd": 0,
            "loss_za_mmd": 0,
            "loss_z_info": 0,
            "loss_zq_info": 0,
            "loss_za_info": 0,
            "loss_qa_info": 0,
        }
        self.cnt_steps = 0

    def train(self, c_ids, q_ids, a_ids, start_positions, end_positions):
        self.vae.train()

        # Forward
        return_dict = self.vae(c_ids, q_ids, a_ids, start_positions, end_positions)

        # Backward
        self.optimizer.zero_grad()
        return_dict["total_loss"].backward()
        # Step
        self.optimizer.step()

        for key, value in return_dict.items():
            self.losses[key] += value

        self.cnt_steps += 1
        if self.cnt_steps % 100 == 0:
            self.print_log()


    def print_log(self, log_type="step", epoch=None):
        """
            log_type: "epoch" or "step"
            epoch: enter a number
        """
        assert log_type in ["step", "epoch"]
        assert log_type == "step" or (log_type == "epoch" and epoch is not None)

        log_str = ""
        if log_type == "step":
            log_str = "\nStep={:d} - ".format(self.cnt_steps)
        else:
            log_str = "\nEpoch={:d} - ".format(epoch)

        for key, value in self.losses.items():
            log_str += "{:s}={:.4f};".format(key, value / self.cnt_steps)

        print(log_str + "\n")


    def _reset_loss_values(self):
        for key in self.losses.keys():
            self.losses[key] = 0


    def reset_cnt_steps(self):
        self.cnt_steps = 0
        self._reset_loss_values()


    def generate_posterior(self, c_ids, q_ids, a_ids):
        with torch.no_grad():
            zq, za = self.vae.posterior_encoder(c_ids, q_ids, a_ids)
            q_ids, start_positions, end_positions = self.vae.generate(zq, za, c_ids)
        return q_ids, start_positions, end_positions, zq


    def generate_answer_logits(self, c_ids, q_ids, a_ids):
        with torch.no_grad():
            zq, za = self.vae.posterior_encoder(c_ids, q_ids, a_ids)
            start_logits, end_logits = self.vae.return_answer_logits(zq, za, c_ids)
        return start_logits, end_logits


    def generate_prior(self, c_ids):
        with torch.no_grad():
            zq, za = self.vae.prior_encoder(c_ids)
            q_ids, start_positions, end_positions = self.vae.generate(zq, za, c_ids)
        return q_ids, start_positions, end_positions, zq    


    def save(self, filename):
        params = {
            'vae_state_dict': self.vae.state_dict(),
            'args': self.args
        }
        torch.save(params, filename)


    def load_model_state_dict(self, filename):
        params = torch.load(filename)
        self.vae.load_state_dict(params["vae_state_dict"])


    def set_eval_mode(self, enable=True):
        if enable:
            self.vae.eval()