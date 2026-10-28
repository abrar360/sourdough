# sourdough

Starter code for PyTorch experiments

## Instructions

This repo is an example template that provides a format for converting a working notebook or existing codebase into a lightweight and organized toolkit for rapid experimentation and prototyping.

The reference code should be present in the /REFERENCE folder. This may consist of a jupyter notebook or an official implementation repository released by the authors of a research paper.

A conv autoencoder (convae) is used as an example throughout.

The template is organized as follows:

```
train.py            wrapper to kick off training: python train.py --config configs/base.yaml name=x lr=3e-4 latent=8
sweep.py            wrapper to kick off sweeps: python sweep.py configs/sweeps/example.yaml [--dry-run] [--only a,b] [key=value ...]
convae/config.py    all configurable parameters
convae/data.py      dataloaders and preprocessing
convae/model.py     model architecture
convae/losses.py    loss functions
convae/viz.py       plotting and visualization utils
convae/train.py     training loop
tests/smoke_test.py sanity checks
```

You should replace "convae" with whatever type of model you are implementing.
For example, if you are working with flow matching then that folder should be named "flow-matching".

Fill out the template in the following order:

1. data.py
2. model.py
3. losses.py
4. vis.py
5. train.py
6. config.py
7. smoke_test.py

Try to optimize performance for a 4090 card.
- for example, things like torch.compile, bfloat16, etc.

Once you complete these, then carefully test the code using the following procedure:
1. run the smoke test.
2. attempt a small end to end training job on mock data or a tiny subset of data
3. do one final check to make sure that your implementation is a faithful reproduction of the reference notebook or existing codebase you are porting.

