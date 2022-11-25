import torch
import torch.nn as nn

from models import DiscreteVAE, InfoMaxModel


class VAETrainer(object):
    def __init__(self, args):
        self.args = args
        self.clip = args.clip
        self.device = args.device

        huggingface_model = args.huggingface_model
        if "large" in huggingface_model:
            emsize = 1024
        else:
            emsize = 768

        self.vae = DiscreteVAE(args).to(self.device)
        params = filter(lambda p: p.requires_grad, self.vae.parameters())
        self.optimizer_vae = torch.optim.Adam(params, lr=args.lr)

        self.lambda_z_info = args.lambda_z_info
        if self.lambda_z_info > 0:
            self.stop_infomax_opt = False

            self.embedding = self.vae.posterior_encoder.embedding

            self.q_init_dim = args.dec_q_nlayers*args.dec_q_nhidden
            self.q_infomax_net = InfoMaxModel(self.q_init_dim*2, emsize*2).to(self.device)
            # q_info_params = filter(lambda p: p.requires_grad, self.q_infomax_net.parameters())
            self.optimizer_q_infomax = torch.optim.Adam(self.q_infomax_net.parameters(), lr=args.lr_infomax)

            self.a_init_dim = emsize
            self.a_infomax_net = InfoMaxModel(self.a_init_dim, emsize*2).to(self.device)
            # a_info_params = filter(lambda p: p.requires_grad, self.a_infomax_net.parameters())
            self.optimizer_a_infomax = torch.optim.Adam(self.a_infomax_net.parameters(), lr=args.lr_infomax)

        self.total_loss = 0
        self.loss_q_rec = 0
        self.loss_a_rec = 0
        self.loss_zq_kl = 0
        self.loss_za_kl = 0
        self.loss_zq_info = 0
        self.loss_za_info = 0
        # self.loss_prior_info = 0
        self.loss_qa_info = 0
        self.cnt_steps = 0

    def train(self, c_ids, q_ids, a_ids, start_positions, end_positions):
        self.vae = self.vae.train()
        if self.lambda_z_info > 0 and not self.stop_infomax_opt:
            self.a_infomax_net = self.a_infomax_net.train()
            self.q_infomax_net = self.q_infomax_net.train()

        # Forward
        loss, \
        loss_q_rec, loss_a_rec, \
        loss_zq_kl, loss_za_kl, \
        loss_qa_info, latent_vars \
        = self.vae(c_ids, q_ids, a_ids, start_positions, end_positions)

        loss_zq_info, loss_za_info = torch.tensor(0), torch.tensor(0)
        if self.lambda_z_info > 0:
            q_embeddings = self.embedding(q_ids).mean(dim=1)
            c_embeddings = self.embedding(c_ids).mean(dim=1)
            c_a_embeddings = self.embedding(c_ids, a_ids, None).sum(dim=-1).div(a_ids.sum(dim=-1, keepdim=True))
            posterior_zq, prior_zq, posterior_za, prior_za = latent_vars
            posterior_zq_init, posterior_za_init = self.vae.return_init_state(posterior_zq, posterior_za)
            prior_zq_init, prior_za_init = self.vae.return_init_state(prior_zq, prior_za)
            posterior_zq_init = torch.cat((posterior_zq_init[0].transpose(0, 1).contiguous().view(-1, self.q_init_dim),
                        posterior_zq_init[1].transpose(0, 1).contiguous().view(-1, self.q_init_dim)), dim=-1)
            prior_zq_init = torch.cat((prior_zq_init[0].transpose(0, 1).contiguous().view(-1, self.q_init_dim),
                        prior_zq_init[1].transpose(0, 1).contiguous().view(-1, self.q_init_dim)), dim=-1)

            loss_zq_info = 0.5*(self.q_infomax_net(torch.cat((q_embeddings, c_embeddings), dim=-1), posterior_zq_init) \
                    + self.q_infomax_net(torch.cat((q_embeddings, c_embeddings), dim=-1), prior_zq_init))
            loss += self.lambda_z_info * loss_zq_info

            print(c_a_embeddings.size())
            loss_za_info = 0.5*(self.a_infomax_net(torch.cat((c_a_embeddings, c_embeddings), dim=-1), posterior_za_init) \
                           + self.a_infomax_net(torch.cat((c_a_embeddings, c_embeddings), dim=-1), prior_za_init))
            loss += self.lambda_z_info * loss_za_info

        # Backward
        self.optimizer_vae.zero_grad()
        loss.backward(retain_graph=True)
        # Step
        self.optimizer_vae.step()

        if self.lambda_z_info > 0 and not self.stop_infomax_opt:
            self.optimizer_q_infomax.zero_grad()
            loss_zq_info.backward(inputs=list(self.q_infomax_net.parameters()))
            self.optimizer_q_infomax.step()

            self.optimizer_a_infomax.zero_grad()
            loss_za_info.backward(inputs=list(self.a_infomax_net.parameters()))
            self.optimizer_a_infomax.step()

        self.total_loss += loss.item()
        self.loss_q_rec += loss_q_rec.item()
        self.loss_a_rec += loss_a_rec.item()
        self.loss_zq_kl += loss_zq_kl.item()
        self.loss_za_kl += loss_za_kl.item()
        self.loss_zq_info += loss_zq_info.item()
        self.loss_za_info += loss_za_info.item()
        # self.loss_prior_info += loss_prior_info.item()
        self.loss_qa_info += loss_qa_info.item()

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
            log_str = "\nStep={:d} - AVG LOSS={:.4f} (q_rec={:.4f}, a_rec={:.4f}, zq_kl={:.4f}, za_kl={:.4f}, zq_info={:.4f}, za_info={:.4f}, qa_info={:.4f})"
        else:
            log_str = "\nEpoch stats (step={:d}) - AVG LOSS={:.4f} (q_rec={:.4f}, a_rec={:.4f}, zq_kl={:.4f}, za_kl={:.4f}, zq_info={:.4f}, za_info={:.4f}, qa_info={:.4f})"

        log_str = log_str.format(self.cnt_steps, float(self.total_loss / self.cnt_steps), float(self.loss_q_rec / self.cnt_steps),
                    float(self.loss_a_rec / self.cnt_steps), float(self.loss_zq_kl / self.cnt_steps), float(self.loss_za_kl / self.cnt_steps),
                    float(self.loss_zq_info / self.cnt_steps), float(self.loss_za_info / self.cnt_steps), float(self.loss_qa_info / self.cnt_steps))
        print(log_str)

    def _reset_loss_values(self):
        self.total_loss = 0
        self.loss_q_rec = 0
        self.loss_a_rec = 0
        self.loss_zq_kl = 0
        self.loss_za_kl = 0
        self.loss_zq_info = 0
        self.loss_za_info = 0
        self.loss_qa_info = 0

    def stop_infomax_optimization(self):
        self.stop_infomax_opt = True
        self.q_infomax_net = self.q_infomax_net.eval()
        self.a_infomax_net = self.a_infomax_net.eval()

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
        if self.lambda_z_info > 0:
            params = {
                'vae_state_dict': self.vae.state_dict(),
                'q_info_state_dict': self.q_infomax_net.state_dict(),
                'a_info_state_dict': self.a_infomax_net.state_dict(),
                'args': self.args
            }
        else:
            params = {
                'vae_state_dict': self.vae.state_dict(),
                'args': self.args
            }
        torch.save(params, filename)

    def load_model_state_dict(self, filename, load_infomax_model=False):
        params = torch.load(filename)
        self.vae.load_state_dict(params["vae_state_dict"])
        if load_infomax_model:
            self.q_infomax_net.load_state_dict(params["q_info_state_dict"])
            self.a_infomax_net.load_state_dict(params["a_info_state_dict"])