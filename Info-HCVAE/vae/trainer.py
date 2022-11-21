import torch
import torch.nn as nn

from models import DiscreteVAE, return_mask_lengths


class VAETrainer(object):
    def __init__(self, args):
        self.args = args
        self.clip = args.clip
        self.device = args.device

        self.vae = DiscreteVAE(args, use_custom_embeddings=args.use_custom_embeddings_impl).to(self.device)
        params = filter(lambda p: p.requires_grad, self.vae.parameters())
        self.optimizer = torch.optim.Adam(params, lr=args.lr)

        self.total_loss = 0
        self.loss_q_rec = 0
        self.loss_a_rec = 0
        self.loss_zq_kl = 0
        self.loss_za_kl = 0
        self.loss_zq_mmd = 0
        self.loss_za_mmd = 0
        self.loss_prior_info = 0
        self.loss_info = 0
        self.cnt_steps = 0

    def train(self, c_ids, q_ids, a_ids, start_positions, end_positions):
        self.vae = self.vae.train()

        # Forward
        loss, \
        loss_q_rec, loss_a_rec, \
        loss_zq_kl, loss_za_kl, \
        loss_zq_mmd, loss_za_mmd, \
        loss_prior_info, loss_info \
        = self.vae(c_ids, q_ids, a_ids, start_positions, end_positions)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Step
        self.optimizer.step()

        self.total_loss += loss.item()
        self.loss_q_rec += loss_q_rec.item()
        self.loss_a_rec += loss_a_rec.item()
        self.loss_zq_kl += loss_zq_kl.item()
        self.loss_za_kl += loss_za_kl.item()
        self.loss_zq_mmd += loss_zq_mmd.item()
        self.loss_za_mmd += loss_za_mmd.item()
        self.loss_prior_info += loss_prior_info.item()
        self.loss_info += loss_info.item()

        self.cnt_steps += 1
        if self.cnt_steps % 100 == 0:
            log_str = "\nStep={:d} - AVG LOSS={:.4f} (q_rec={:.4f}, a_rec={:.4f}, zq_kl={:.4f}, za_kl={:.4f}, zq_mmd={:.4f}, \
                za_mmd={:.4f}, prior_info={:4f}, info={:.4f})"
            log_str = log_str.format(self.cnt_steps, float(self.total_loss / self.cnt_steps), float(self.loss_q_rec / self.cnt_steps),
                        float(self.loss_a_rec / self.cnt_steps), float(self.loss_zq_kl / self.cnt_steps), float(self.loss_za_kl / self.cnt_steps),
                        float(self.loss_zq_mmd / self.cnt_steps), float(self.loss_za_mmd / self.cnt_steps), float(self.loss_prior_info / self.cnt_steps),
                        float(self.loss_info / self.cnt_steps))
            print(log_str)

    def _reset_loss_values(self):
        self.total_loss = 0
        self.loss_q_rec = 0
        self.loss_a_rec = 0
        self.loss_zq_kl = 0
        self.loss_za_kl = 0
        self.loss_zq_mmd = 0
        self.loss_za_mmd = 0
        self.loss_info = 0

    def reset_cnt_steps(self):
        self.cnt_steps = 0
        self._reset_loss_values()

    def generate_posterior(self, c_ids, q_ids, a_ids):
        self.vae = self.vae.eval()
        with torch.no_grad():
            _, _, zq, _, za = self.vae.posterior_encoder(c_ids, q_ids, a_ids)
            q_ids, start_positions, end_positions = self.vae.generate(zq, za, c_ids)
        return q_ids, start_positions, end_positions, zq

    def generate_answer_logits(self, c_ids, q_ids, a_ids):
        self.vae = self.vae.eval()
        with torch.no_grad():
            _, _, zq, _, za = self.vae.posterior_encoder(c_ids, q_ids, a_ids)
            start_logits, end_logits = self.vae.return_answer_logits(zq, za, c_ids)
        return start_logits, end_logits

    def generate_prior(self, c_ids):
        self.vae = self.vae.eval()
        with torch.no_grad():
            _, _, zq, _, za = self.vae.prior_encoder(c_ids)
            q_ids, start_positions, end_positions = self.vae.generate(zq, za, c_ids)
        return q_ids, start_positions, end_positions, zq    

    def save(self, filename):
        params = {
            'state_dict': self.vae.state_dict(),
            'args': self.args
        }
        torch.save(params, filename)


    def load_model_state_dict(self, filename):
        params = torch.load(filename)
        self.vae.load_state_dict(params["state_dict"])