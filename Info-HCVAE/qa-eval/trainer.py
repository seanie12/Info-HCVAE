import collections
import os
import time
import json
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AdamW
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_

from squad_utils import write_predictions, evaluate

from utils import eta, progress_bar, user_friendly_time, time_since

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.set_random_seed(random_seed=args.random_seed)
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model)


    def make_model_env_no_dist(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.args.bert_model).to(device)
        self.get_dev_loader()
        self.get_test_loader()

        self.args.batch_size = int(self.args.batch_size)
        self.get_pretrain_loader()

        self.pretrain_t_total = len(self.pretrain_loader) * self.args.pretrain_epochs

        no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = AdamW(self.optimizer_grouped_parameters,
        lr=self.args.pretrain_lr, eps=self.args.adam_epsilon)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
        num_warmup_steps=self.args.warmup_steps, num_training_steps=self.pretrain_t_total)


    def make_model_env(self, gpu, ngpus_per_node):
        if gpu is not None:
            self.args.gpu = self.args.devices[gpu]

        if self.args.use_cuda and self.args.distributed:
            if self.args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                self.args.rank = self.args.rank * ngpus_per_node + gpu
            dist.init_process_group(backend=self.args.dist_backend, init_method=self.args.dist_url,
                                    world_size=self.args.world_size, rank=self.args.rank)

        if self.args.rank == 0:
            if self.args.debug:
                print("debugging mode on.")

        self.model = AutoModelForQuestionAnswering.from_pretrained(self.args.bert_model)
        if self.args.rank == 0:
            self.get_dev_loader()
            self.get_test_loader()

        self.args.batch_size = int(self.args.batch_size / ngpus_per_node)
        self.get_pretrain_loader()

        self.pretrain_t_total = len(self.pretrain_loader) * self.args.pretrain_epochs

        no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = AdamW(self.optimizer_grouped_parameters,
        lr=self.args.pretrain_lr, eps=self.args.adam_epsilon)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
        num_warmup_steps=self.args.warmup_steps, num_training_steps=self.pretrain_t_total)

        if self.args.use_cuda:
            torch.cuda.set_device(self.args.gpu)
            self.model.cuda(self.args.gpu)
            self.args.workers = int((self.args.workers + ngpus_per_node - 1) / ngpus_per_node)
            self.model = DistributedDataParallel(self.model, device_ids=[self.args.gpu],
                                                 find_unused_parameters=True)

        cudnn.benchmark = True

    def get_pretrain_loader(self):
        data = self.args.pretrain_dataset

        # self.pretrain_sampler = DistributedSampler(data)
        # self.pretrain_loader = DataLoader(data, num_workers=self.args.workers, pin_memory=True,
        #                                   sampler=self.pretrain_sampler, batch_size=self.args.batch_size)
        self.pretrain_loader = DataLoader(data, batch_size=self.args.batch_size)


    def get_dev_loader(self):
        data = self.args.dev_dataset

        self.dev_loader = DataLoader(data, shuffle=False, batch_size=self.args.batch_size)
        self.dev_examples = self.args.dev_examples
        self.dev_features = self.args.dev_features

    def get_test_loader(self):
        data = self.args.test_dataset

        self.test_loader = DataLoader(data, shuffle=False, batch_size=self.args.batch_size)
        self.test_examples = self.args.test_examples
        self.test_features = self.args.test_features

    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        os.makedirs(self.args.model_save_path, exist_ok=True)

        if self.args.ckpt_path is not None:
            self.model.load_state_dict(torch.load(self.args.ckpt_path))

        self.model.zero_grad()

        for epoch in range(0, self.args.pretrain_epochs):
            if epoch+1 < self.args.resume_epochs:
                continue

            num_batches = len(self.pretrain_loader)
            # self.pretrain_sampler.set_epoch(epoch)
            start = time.time()

            msg = ""
            running_loss, epoch_loss = 0, 0
            # pretrain with unsupervised dataset
            for step, batch in enumerate(self.pretrain_loader, start=1):
                if step <= self.args.resume_steps:
                    continue

                self.model.train()
                input_ids, input_mask, seg_ids, start_positions, end_positions = batch

                seq_len = torch.sum(torch.sign(input_ids), 1)
                max_len = torch.max(seq_len)
                input_ids = input_ids[:, :max_len].clone().to(device)
                input_mask = input_mask[:, :max_len].clone().to(device)
                seg_ids = seg_ids[:, :max_len].clone().to(device)
                start_positions = start_positions.clone().to(device)
                end_positions = end_positions.clone().to(device)

                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": input_mask,
                    "token_type_ids": seg_ids,
                    "start_positions": start_positions,
                    "end_positions": end_positions
                }
                loss = self.model(**inputs)[0]
                loss.backward()

                running_loss += loss.item()
                epoch_loss += loss.item()

                clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

                if step % 500 == 0: # self.args.rank == 0:
                    msg = "PRETRAIN {}/{} {} - ETA : {} - LOSS : {:.4f}".format(step,
                                num_batches, progress_bar(step, num_batches),
                                eta(start, step, num_batches),
                                float(running_loss / 500))
                    print(msg, end="\r")
                    running_loss = 0

                if (step + 1) % 10000 == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.args.model_save_path, "bert-qa-step-{:07d}-epoch-{:02d}.pt".format(step, epoch)))

                if self.args.debug:
                    break

            # print log & save model
            # if self.args.rank == 0:
            result_dict = self.evaluate_model(msg)
            em = result_dict["exact_match"]
            f1 = result_dict["f1"]
            print("\nPRETRAIN took {} LOSS = {:.4f} / DEV - F1: {:.4f}, EM: {:.4f}\n"
                    .format(user_friendly_time(time_since(start)), epoch_loss / len(self.pretrain_loader), f1, em))

            torch.save(self.model.state_dict(), os.path.join(self.args.model_save_path, "bert-qa-epoch-{:02d}.pt".format(epoch)))

        # if self.args.rank == 0:

        result_dict = self.evaluate_model("TEST", False)
        em = result_dict["exact_match"]
        f1 = result_dict["f1"]
        print("\nFINAL TEST - F1: {:.4f}, EM: {:.4f}\n"
                .format(f1, em))

    def evaluate_model(self, msg, dev=True):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if dev:
            eval_examples = self.dev_examples
            eval_features = self.dev_features
            eval_loader = self.dev_loader
            eval_file = self.args.dev_json_file
        else:
            eval_examples = self.test_examples
            eval_features = self.test_features
            eval_loader = self.test_loader
            eval_file = self.args.test_json_file

        RawResult = collections.namedtuple("RawResult",
                                           ["unique_id", "start_logits", "end_logits"])
        all_results = []
        example_index = -1
        self.model.eval()
        num_val_batches = len(eval_loader)
        for i, batch in enumerate(eval_loader):
            input_ids, input_mask, seg_ids = batch
            seq_len = torch.sum(torch.sign(input_ids), 1)
            max_len = torch.max(seq_len)

            input_ids = input_ids[:, :max_len].clone().to(device)
            input_mask = input_mask[:, :max_len].clone().to(device)
            seg_ids = seg_ids[:, :max_len].clone().to(device)

            with torch.no_grad():
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": input_mask,
                    "token_type_ids": seg_ids
                }
                if hasattr(self.model, 'module'):
                    outputs = self.model.module(**inputs)
                else:
                    outputs = self.model(**inputs)
                batch_start_logits, batch_end_logits = outputs[0], outputs[1]
                batch_size = batch_start_logits.size(0)
            for j in range(batch_size):
                example_index += 1
                start_logits = batch_start_logits[j].detach().cpu().tolist()
                end_logits = batch_end_logits[j].detach().cpu().tolist()
                eval_feature = eval_features[example_index]
                unique_id = int(eval_feature.unique_id)
                all_results.append(RawResult(unique_id=unique_id,
                                             start_logits=start_logits,
                                             end_logits=end_logits))
            msg2 = "{} => Evaluating :{}/{}".format(msg, i, num_val_batches)
            print(msg2, end="\r")

        os.makedirs("./save/", exist_ok=True)
        output_prediction_file = os.path.join("./save/", "prediction.json")
        write_predictions(eval_examples, eval_features, all_results,
                          n_best_size=20, max_answer_length=30, do_lower_case=True,
                          output_prediction_file=output_prediction_file,
                          verbose_logging=False,
                          version_2_with_negative=False,
                          null_score_diff_threshold=0,
                          noq_position=False)

        with open(output_prediction_file) as prediction_file:
            predictions = json.load(prediction_file)

        with open(eval_file) as f:
            data_json = json.load(f)
            dataset = data_json["data"]

        result_dict = evaluate(dataset, predictions)
        
        return result_dict

    def set_random_seed(self, random_seed=2019):
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)
