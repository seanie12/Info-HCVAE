# Generating Diverse and Consistent QA pairs from Contexts with Information-Maximizing Hierarchical Conditional VAEs
This is the **Pytorch implementation** for the paper Generating Diverse and Consistent QA pairs from Contexts with
Information-Maximizing Hierarchical Conditional VAEs (**ACL 2020**, **long paper**) :
[[Paper]](https://www.aclweb.org/anthology/2020.acl-main.20/) [[Slide]](https://drive.google.com/file/d/17oakiVKIaQ1Y_hSCkGfUIp8P6ORSYjjz/view?usp=sharing) [[Video]](https://slideslive.com/38928851/generating-diverse-and-consistent-qa-pairs-from-contexts-with-informationmaximizing-hierarchical-conditional-vaes).



## Abstract
<img align="middle" width="800" src="https://github.com/seanie12/Info-HCVAE/blob/master/images/concept.png">
One of the most crucial challenges in question answering (QA) is the scarcity of labeled data, since it is costly to obtain question-answer (QA) pairs for a target text domain with human annotation. An alternative approach to
tackle the problem is to use automatically generated QA pairs from either the problem context or from large amount of unstructured texts (e.g. Wikipedia). In this work, we propose a hierarchical conditional variational autoencoder
(HCVAE) for generating QA pairs given unstructured texts as contexts, while maximizing
the mutual information between generated QA pairs to ensure their consistency. We validate
our Information Maximizing Hierarchical Conditional Variational AutoEncoder (InfoHCVAE) on several benchmark datasets by
evaluating the performance of the QA model (BERT-base) using only the generated QA pairs (QA-based evaluation) or by using both the generated and human-labeled pairs (semisupervised learning) for training, against stateof-the-art baseline models. The results show that our model obtains impressive performance gains over all baselines on both tasks,
using only a fraction of data for training.

__Contribution of this work__
- We propose a novel hierarchical variational framework for generating diverse QA pairs from a single context, which is, to our knowledge, the first probabilistic generative model for questionanswer pair generation (QAG). 
- We propose an InfoMax regularizer which effectively enforces the consistency between the
generated QA pairs, by maximizing their mutual information. This is a novel approach in resolving consistency between QA pairs for QAG.
- We evaluate our framework on several benchmark datasets by either training a new model entirely using generated QA pairs (QA-based evaluation), or use both ground-truth and generated QA pairs (semi-supervised QA). Our model
achieves impressive performances on both tasks, largely outperforming existing QAG baselines.


## Dependencies
This code is written in Python. Dependencies include
* python >= 3.6
* pytorch >= 1.4
* json-lines
* tqdm
* [pytorch_scatter](https://github.com/rusty1s/pytorch_scatter)
* [transfomers](https://github.com/huggingface/transformers)


## Download data
```bash
mkdir data/squad 
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O ./data/squad/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O ./data/squad/dev-v1.1.json
```



## Train Info-HCVAE
Train Info-HCVAE with the following command. The checkpoint will be save at ./save/vae-checkpoint.
```bash
cd vae
python main.py
```
## Generate QA pairs 
Generate QA pairs from unlabeled paragraphs. If you generate QA pairs from SQuAD, use option --squad.
```bash
cd vae
python translate.py --data_file "DATA DIRECTORY for paragraph" --checkpoint "directory for Info-HCVAE model" --output_file "output file directory" --k "the number of QA pairs to sample for each paragraph" --ratio "the percentage of context to use"
```


## QA-based-Evaluation (QAE) & Reverse-QA-based-Evaluation (R-QAE)
```bash
cd qa-eval
python main.py --devices 0_1_2 --pretrain_file $PATH_TO_qaeval --unlabel_ratio 0.1 --lazy_loader --batch_size 24
```

## Semi-Supervised Learning
TBA
```bash
cd qa-eval
python main.py --devices 0_1_2_3 --pretrain_file $PATH_TO_semieval --unlabel_ratio 1.0 --lazy_loader --batch_size 32
```

## Pretrained Model & Synthetic QA pairs
TBA


## Reference
To cite the code/data/paper, please use this BibTex
```bibtex
@inproceedings{lee2020generating,
  title={Generating Diverse and Consistent QA pairs from Contexts with Information-Maximizing Hierarchical Conditional VAEs},
  author={Lee, Dong Bok and Lee, Seanie and Jeong, Woo Tae and Kim, Donghwan and Hwang, Sung Ju},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020}
}
```
