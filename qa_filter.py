import argparse
import torch
import torch.nn.functional as F
from typing import Final
from transformers import pipeline, BertForQuestionAnswering, BertTokenizer
from infohcvae.datasets import HarvestingQADatasetH5
from torch.utils.data import DataLoader


def _normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _compute_f1(prediction, truth):
    pred_tokens = _normalize_text(prediction).split()
    truth_tokens = _normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)


class SyntheticQaFiltering:
    qa_model_path: Final[str] = "csarron/bert-base-uncased-squad-v1"
    qa_model: Final[BertForQuestionAnswering] = BertForQuestionAnswering.from_pretrained(qa_model_path)
    tokenizer: Final[BertTokenizer] = BertTokenizer.from_pretrained(qa_model_path)
    qa_pipeline: Final[object] = pipeline("question-answering", model=qa_model_path, tokenizer=qa_model_path)

    @staticmethod
    def get_answers_from_positions(input_ids, start_positions, end_positions):
        """
            input_ids: torch.Tensor of shape = (N, seq_len)
            start_positions, end_positions: torch.Tensor of shape = (N,)
        """
        batch_size, _ = input_ids.size()
        answers = []
        for i in range(batch_size):
            answer_start, answer_end = start_positions[i], end_positions[i]
            answer = SyntheticQaFiltering.tokenizer.convert_tokens_to_string(
                SyntheticQaFiltering.tokenizer.convert_ids_to_tokens(input_ids[i][answer_start:answer_end]))
            answers.append(answer)

        return answers

    @staticmethod
    def get_qa_model_prediction(input_ids, input_mask, segment_ids):
        """
            input_ids, input_mask, segment_ids: torch.Tensor with shape = (N, seq_len),
                the `input_ids` contains question & context
        """
        outputs = SyntheticQaFiltering.qa_model(input_ids=input_ids, attention_mask=input_mask,
                                                token_type_ids=segment_ids)
        start_logits, end_logits = outputs[1], outputs[2]

        mask = torch.matmul(input_mask.unsqueeze(2).float(),
                            input_mask.unsqueeze(1).float())
        mask = torch.triu(mask) == 0
        score = (F.log_softmax(start_logits, dim=1).unsqueeze(2)
                 + F.log_softmax(end_logits, dim=1).unsqueeze(1))
        score = score.masked_fill(mask, -10000.0)
        score, start_positions = score.max(dim=1)
        score, end_positions = score.max(dim=1)
        start_positions = torch.gather(start_positions,
                                       1,
                                       end_positions.view(-1, 1)).squeeze(1)

        start_positions = start_positions.unsqueeze(1)
        end_positions = end_positions.unsqueeze(1)

        return SyntheticQaFiltering.get_answers_from_positions(input_ids, start_positions, end_positions), \
            start_positions, end_positions

    @staticmethod
    def filter_qa(gen_qa_dataset: HarvestingQADatasetH5, batch_size: int = 256):
        dataloader = DataLoader(gen_qa_dataset, batch_size=batch_size)
        qa_idx = 0
        for step, batch in enumerate(dataloader, start=1):
            input_ids, input_mask, seg_ids, start_positions, end_positions = batch
            qa_model_answers, qa_model_start_positions, qa_model_end_positions = \
                SyntheticQaFiltering.get_qa_model_prediction(input_ids, input_mask, seg_ids)
            synthetic_answers = SyntheticQaFiltering.get_answers_from_positions(input_ids, start_positions,
                                                                                end_positions)
            for i in range(batch_size):
                f1_score = _compute_f1(synthetic_answers[i], qa_model_answers[i])
                if f1_score < 40.0:
                    gen_qa_dataset.rewrite_start_end_positions(qa_idx + i, qa_model_start_positions[i],
                                                               qa_model_end_positions[i])

            qa_idx += batch_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_file", default="./data/harv_synthetic_data_semi/0.4_replaced_1.0_harv_features.h5",
                        type=str, help="path of training data file, only .h5 allowed")

    args = parser.parse_args()

    dataset = HarvestingQADatasetH5(args.pretrain_file, mode="a")
    SyntheticQaFiltering.filter_qa(dataset)
