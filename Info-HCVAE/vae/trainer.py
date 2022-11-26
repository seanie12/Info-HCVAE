import torch
import torch.nn as nn
from model.qag_vae import DiscreteVAE

class VAETrainer(object):
    def __init__(self, args):
        self.args = args
        self.clip = args.clip
        self.device = args.device

        self.vae = DiscreteVAE(args).to(self.device)
        self.optimizer = torch.optim.SGD(self.vae.get_par, lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)

        self.total_loss = 0
        self.loss_q_rec = 0
        self.loss_a_rec = 0
        self.loss_zq_kl = 0
        self.loss_za_kl = 0
        self.loss_zq_mmd = 0
        self.loss_za_mmd = 0
        self.loss_qa_info = 0
        self.cnt_steps = 0

    def train(self, c_ids, q_ids, a_ids, start_positions, end_positions):
        self.vae.train()

        # Forward
        return_dict = self.vae(c_ids, q_ids, a_ids, start_positions, end_positions)
        total_loss = return_dict["total_loss"]

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        # Step
        self.optimizer.step()

        self.total_loss += total_loss.item()
        self.loss_q_rec += return_dict["loss_q_rec"].item()
        self.loss_a_rec += return_dict["loss_a_rec"].item()
        self.loss_zq_kl += return_dict["loss_zq_kl"].item()
        self.loss_za_kl += return_dict["loss_za_kl"].item()
        self.loss_zq_mmd += return_dict["loss_zq_mmd"].item()
        self.loss_za_mmd += return_dict["loss_za_mmd"].item()
        self.loss_qa_info += return_dict["loss_qa_info"].item()

        self.cnt_steps += 1
        if self.cnt_steps % 100 == 0:
            self.print_log()


    def print_log(self, log_type="step"):
        """
            log_type: "epoch" or "step"
        """
        assert log_type in ["step", "epoch"]
        log_str = ""
        if log_type == "step":
            log_str = "\nStep={:d} - AVG LOSS={:.4f} (q_rec={:.4f}, a_rec={:.4f}, zq_kl={:.4f}, za_kl={:.4f}, zq_mmd={:.4f}, za_mmd={:.4f}, qa_info={:.4f})"
        else:
            log_str = "\nEpoch stats (step={:d}) - AVG LOSS={:.4f} (q_rec={:.4f}, a_rec={:.4f}, zq_kl={:.4f}, za_kl={:.4f}, zq_mmd={:.4f}, za_mmd={:.4f}, qa_info={:.4f})"

        log_str = log_str.format(self.cnt_steps, float(self.total_loss / self.cnt_steps), float(self.loss_q_rec / self.cnt_steps),
                    float(self.loss_a_rec / self.cnt_steps), float(self.loss_zq_kl / self.cnt_steps), float(self.loss_za_kl / self.cnt_steps),
                    float(self.loss_zq_mmd / self.cnt_steps), float(self.loss_za_mmd / self.cnt_steps), float(self.loss_qa_info / self.cnt_steps))
        print(log_str)


    def _reset_loss_values(self):
        self.total_loss = 0
        self.loss_q_rec = 0
        self.loss_a_rec = 0
        self.loss_zq_kl = 0
        self.loss_za_kl = 0
        self.loss_zq_mmd = 0
        self.loss_za_mmd = 0
        self.loss_qa_info = 0


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