# Semi-supervised Node Classification

Evaluation scripts for various methods on the Cora, CiteSeer and PubMed citation networks, Amazon datasets as well.
Each experiment is repeated 100 times on either a fixed train/val/test split or on multiple random splits, except for Amazon datasets, they are 'test/rest' split, which is the same in the paper:

* **[GAT](https://arxiv.org/abs/1710.10903)**: `python ori_gat.py`
* **[GATv2](https://arxiv.org/abs/2105.14491)**: `python ori_gatv2.py`


Run the test suite via

```bash
$ python gat.py --dataset [dataname][default=Cora] --runs 1[default=100] --epochs 1[defaults=1000]
```

### Added GATv2 - version 1.0.4

the *kl loss* is mainly added in *train_eval.py*, with a simply addition after F.nll_loss(). 

$$
total\_loss = cl\_loss + \alpha_1 \cdot kl\_loss + \alpha_2 \cdot kl\_loss
$$

In terms of model implementation, we mainly look over *new_gat_layer.py*

We adopt 2 layers' noises in this version, from layer1 and layer2. 
For layer1 noise, we use a simple 2-layer MLP.
We then calculate them both through kl loss, using a percentage alpha setting as a hyper-parameter

For result saving, we add **logger.py**, which can log the results into 'results/' directory

In this version, we implement GATv2 model, and its variant with our kl loss.

### Experiments

|                     |     Actor      |   Chameleon    |    Cornell     |    Squirrel    |     Texas      |   Wisconsin    |
| :-----------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |
|      base-GAT       | 0.2757 ± 0.008 | 0.4355 ± 0.022 | 0.4351 ± 0.071 | 0.2729 ± 0.016 | 0.5865 ± 0.042 | 0.5235 ± 0.026 |
| kl-GAT (maybe best) | 0.2828 ± 0.008 | 0.4386 ± 0.023 | 0.4892 ± 0.052 | 0.2750 ± 0.014 | 0.6297 ± 0.058 | 0.5451 ± 0.040 |

### Settings

- Runs: 10
- Epochs: 1000
- Learning Rate: 0.005
- Weight Decay: 0.0005
- Hidden Dims: 8
- Model Layers: 2
- Multi-heads: 8
