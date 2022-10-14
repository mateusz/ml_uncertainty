#!/bin/bash
set -euo pipefail

dvc exp run -f -n 'base' -S 'generate.r_loc=[-3.0,3.0]' -S 'generate.r_scale=[0.1,1.0]' -S 'train.dense_dropout=0.5'
cp metrics/*.json evaluation/*.png notebooks/results/base
dvc exp run -f -n 'high_loc' -S 'generate.r_loc=[10.0,13.0]' -S 'generate.r_scale=[0.1,1.0]' -S 'train.dense_dropout=0.5'
cp metrics/*.json evaluation/*.png notebooks/results/high_loc
dvc exp run -f -n 'high_scale' -S 'generate.r_loc=[-3.0,3.0]' -S 'generate.r_scale=[1.0,10.0]' -S 'train.dense_dropout=0.5'
cp metrics/*.json evaluation/*.png notebooks/results/high_scale
dvc exp run -f -n 'high_dropout' -S 'generate.r_loc=[-3.0,3.0]' -S 'generate.r_scale=[0.1,1.0]' -S 'train.dense_dropout=0.9'
cp metrics/*.json evaluation/*.png notebooks/results/high_dropout
dvc exp run -f -n 'low_dropout' -S 'generate.r_loc=[-3.0,3.0]' -S 'generate.r_scale=[0.1,1.0]' -S 'train.dense_dropout=0.1'
cp metrics/*.json evaluation/*.png notebooks/results/low_dropout