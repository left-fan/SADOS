# SADOS

This project combines the original GHRN graph backbone with the SADOS
adaptation code from TTT1. The implementation is pure PyTorch and does not
import the modified implementation in `GHRN_Reddit`.

## Method

Source pretraining has two stages:

1. Fit a baseline BWGNN on source train labels. Select its weights using
   source validation AUROC.
2. Predict class probabilities for all nodes without using target labels.
   Post aggregation (PA) subtracts normalized neighbor probabilities from
   each node's probabilities. Each edge receives the inner product of its
   endpoints' PA vectors. Delete the lowest scoring fraction of edges and
   train a **fresh** BWGNN on this pruned graph. This is the GHRN detector.

The BWGNN encoder uses a Bernstein polynomial filter bank of order
`model.num_layers`. The head returns both logits and embeddings, so the
existing SADOS prototype, shifted augmentation, supervised and contrastive
losses can use it during test time adaptation.

The pruned graph and original graph/split signature are saved in the GHRN
checkpoint. Adaptation checks that signature and reloads exactly the saved
graph. With `preserve_undirected: true`, reciprocal edges are removed
together. This is an implementation choice for TTT1's undirected graph
loader; it differs slightly from counting each directed edge independently.

## Setup

Python 3.10+ and a PyTorch build suitable for your GPU are recommended:

```bash
pip install -r requirements.txt
```

No DGL installation is required. `pymetis` is needed for the optional
density split. `data/raw/amazon.mat` is included. The seven other dataset
configs need their corresponding raw graph files.

## Run

From this project directory:

```bash
python -m experiments.pretrain --config configs/amazon.yaml --device cuda
python -m experiments.adapt_sados --config configs/amazon.yaml --device cuda
```

For a brief pipeline check, use `--baseline-epochs 1 --epochs 1` on the
pretraining command and `--epochs 1` on adaptation. These runs do not
represent trained results.

Each TTA round prints test AUROC, AUPRC and F1. Target labels are used only
for reporting, never for PA pruning, gradient updates or checkpoint
selection. The default TTA mode adapts on test nodes.

## Outputs

- `checkpoints/<dataset>/bwgnn_baseline_<split>_seed<seed>.pt`
- `checkpoints/<dataset>/ghrn_source_<split>_seed<seed>.pt`
- `checkpoints/<dataset>/sados_ghrn_tta_<split>_seed<seed>.pt`
- Separate baseline, GHRN and TTA CSV histories in `logs/csv`.

Run `python -m unittest discover -s tests -v` for unit checks.

Original references: [GHRN paper](https://blacksingular.github.io/papers/www23-GHRN.pdf)
and [official implementation](https://github.com/blacksingular/GHRN).
