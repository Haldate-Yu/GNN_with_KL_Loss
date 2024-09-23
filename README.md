# GATs With KL loss for Semi-supervised Node Classification

Evaluation scripts for various methods on the Cora, CiteSeer and PubMed citation networks, Amazon datasets as well.
Each experiment is repeated 100 times on either a fixed train/val/test split or on multiple random splits, except for
Amazon datasets, they are 'test/rest' split, which is the same in the paper:

* **[GAT](https://arxiv.org/abs/1710.10903)**: `python ori_gat.py`
* **[GATv2](https://arxiv.org/abs/2105.14491)**: `python ori_gatv2.py`
* **[superGAT](https://openreview.net/forum?id=Wi5KUNlqWty)**: `python ori_super_gat.py`

Run the test suite via

```bash
$ python gat.py --dataset [dataname][default=Cora] --runs 1[default=100] --epochs 1[defaults=1000]
```

### Added GAT variants - version 1.0.6

the *kl loss* is mainly added in *train_eval.py*, with a simple addition after F.nll_loss().

$$
total\_loss = cl\_loss + \alpha_1 \cdot kl\_loss + \alpha_2 \cdot kl\_loss
$$

In terms of model implementation, we mainly look over *new_gat_layer.py* (pyg version: **2.0.4**)

We adopt 2 layers' noises in this version, from layer1 and layer2.
For layer1 noise, we use a simple 2-layer MLP.
We then calculate them both through kl loss, using a percentage alpha setting as a hyper-parameter

For result saving, we add **logger.py**, which can log the results into 'results/' directory

In this version, we implement GATv2 model, and its variant with our kl loss.

### Experiments

|                       |     Actor      |   Chameleon    |    Cornell     |    Squirrel    |     Texas      |   Wisconsin    |
|:---------------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
|       base-GAT        | 0.2757 ± 0.008 | 0.4355 ± 0.022 | 0.4351 ± 0.071 | 0.2729 ± 0.016 | 0.5865 ± 0.042 | 0.5235 ± 0.026 |
|     kl-GAT (best)     | 0.2830 ± 0.011 | 0.4386 ± 0.023 | 0.4892 ± 0.052 | 0.2750 ± 0.014 | 0.6297 ± 0.058 | 0.5451 ± 0.040 |
|         GATv2         | 0.2778 ± 0.008 | 0.4361 ± 0.027 | 0.4352 ± 0.091 | 0.2735 ± 0.016 | 0.6000 ± 0.040 | 0.5255 ± 0.020 |
|    kl-GATv2 (best)    | 0.2810 ± 0.009 | 0.4371 ± 0.021 | 0.5243 ± 0.046 | 0.2912 ± 0.018 | 0.6262 ± 0.039 | 0.5416 ± 0.048 |
|      superGAT-MX      | 0.2755 ± 0.007 | 0.4365 ± 0.031 | 0.4351 ± 0.061 | 0.2720 ± 0.013 | 0.5865 ± 0.046 | 0.5255 ± 0.024 |
|      superGAT-SD      | 0.2776 ± 0.007 | 0.4362 ± 0.019 | 0.4351 ± 0.075 | 0.2717 ± 0.016 | 0.5946 ± 0.053 | 0.5392 ± 0.025 |
| kl-superGAT-MX (best) | 0.2836 ± 0.008 | 0.4379 ± 0.022 | 0.5027 ± 0.046 | 0.3132 ± 0.011 | 0.6486 ± 0.058 | 0.5431 ± 0.044 |
| kl-superGAT-SD (best) | 0.2839 ± 0.009 | 0.4377 ± 0.023 | 0.5054 ± 0.056 | 0.3133 ± 0.012 | 0.6568 ± 0.064 | 0.5451 ± 0.032 |

### Settings

- Runs: 10
- Epochs: 1000
- Learning Rate: 0.005
- Weight Decay: 0.0005
- Hidden Dims: [8, 16, 32]
- Model Layers: 2
- Multi-heads: 8
