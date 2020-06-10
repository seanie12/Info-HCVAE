# Generating Diverse and Consistent QA pairs from Contexts with Information-Maximizing Hierarchical Conditional VAEs
This is the **Pytorch implementation** for the paper Generating Diverse and Consistent QA pairs from Contexts with
Information-Maximizing Hierarchical Conditional VAEs (**ACL 2020**, **long paper**) : https://arxiv.org/abs/2005.13837



## Abstract
<img align="middle" width="700" src="https://github.com/seanie12/Info-HCVAE/blob/master/images/concept.png">
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

__Graphical Model__
<br />
<img align="left" width="700" src="https://github.com/seanie12/Info-HCVAE/blob/master/images/graphical_model.png">

<br/><br/>
(HCVAE) for generating QA pairs given unstructured texts as contexts, while maximizing
the mutual information between generated QA pairs to ensure their consistency. We validate
our Information Maximizing Hierarchical Conditional Variational AutoEncoder (InfoHCVAE) on several benchmark datasets by
evaluating the performance of the QA model (BERT-base) using only the generated QA pairs (QA-based evaluation) or by using both the generated and human-labeled pairs (semisupervised learning) for training, against stateof-the-art baseline models. The results show that our model obtains impressive performance gains over all baselines on both tasks,
using only a fraction of data for training.
If you have any questions, please email me. {Id}@gmail.com where Id is lsnfamily02
