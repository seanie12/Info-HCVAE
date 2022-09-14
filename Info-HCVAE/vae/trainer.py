import torch
import torch.nn as nn

from models import DiscreteVAE, return_mask_lengths


class VAETrainer(object):
    def __init__(self, args):
        self.args = args
        self.clip = args.clip
        self.device = args.device

        self.vae = DiscreteVAE(args).to(self.device)
        params = filter(lambda p: p.requires_grad, self.vae.parameters())
        self.optimizer = torch.optim.Adam(params, lr=args.lr)

        self.loss_q_rec = 0
        self.loss_a_rec = 0
        self.loss_zq_kl = 0
        self.loss_za_kl = 0
        self.loss_info = 0

    def train(self, c_ids, q_ids, a_ids, start_positions, end_positions):
        self.vae = self.vae.train()

        # Forward
        loss, \
        loss_q_rec, loss_a_rec, \
        loss_zq_kl, loss_za_kl, \
        loss_info \
        = self.vae(c_ids, q_ids, a_ids, start_positions, end_positions)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Step
        self.optimizer.step()

        self.loss_q_rec = loss_q_rec.item()
        self.loss_a_rec = loss_a_rec.item()
        self.loss_zq_kl = loss_zq_kl.item()
        self.loss_za_kl = loss_za_kl.item()
        self.loss_info = loss_info.item()

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