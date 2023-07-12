# Hawkes Process with Flexible Triggering Kernels

This repository holds the code for the paper: Isik *et al*. Flexible Triggering Kernels for Hawkes Process Modeling. Machine Learning for Healthcare 2023.

## Abstract

Recently proposed encoder-decoder structures for modeling Hawkes processes use transformer-inspired ar-
chitectures, which encode the history of events via embeddings and self-attention mechanisms. These models
deliver better prediction and goodness-of-fit than their RNN-based counterparts. However, they often re-
quire high computational and memory complexity and fail to adequately capture the triggering function
of the underlying process. So motivated, we introduce an efficient and general encoding of the historical
event sequence by replacing the complex (multilayered) attention structures with triggering kernels of the
observed data. Noting the similarity between the triggering kernels of a point process and the attention
scores, we use a triggering kernel to replace the weights used to build history representations. Our estimator
for the triggering function is equipped with a sigmoid gating mechanism that captures local-in-time trigger-
ing effects that are otherwise challenging with standard decaying-over-time kernels. Further, taking both
event type representations and temporal embeddings as inputs, the model learns the underlying triggering
type-time kernel parameters given pairs of event types. We present experiments on synthetic and real data
sets widely used by competing models, and further include a COVID-19 dataset to illustrate the use of
longitudinal covariates. Our results show the proposed model outperforms existing approaches, is more
efficient in terms of computational complexity, and yields interpretable results via direct application of the
newly introduced kernel.

## Usage

```
sh ./run_model.sh
```

## Data

[Processed datasets](https://drive.google.com/drive/folders/1at46RBHZLjpKeKWBRQLgKLIHQcv7Pbdk)

## Reference

```
@inproceedings{isik2023flexible,
  title={Flexible Triggering Kernels for Hawkes Process Modeling},
  author={Isik, Yamac Alican and Davis, Connor and Chapfuwa, Paidamoyo and Henao, Ricardo},
  booktitle={Proceedings of the Machine Learning for Healthcare Conference},
  year={2023},
  organization = {PMLR}
}
```