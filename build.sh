#!/bin/bash
set -euo pipefail

dvc exp run -f -n 'base' -S 'generate.r_loc=[-3.0,3.0]' -S 'generate.r_scale=[0.1,1.0]' -S 'train.dense_dropout=0.5' -S 'train.dense_size=64'
cp metrics/*.json evaluation/*.png results/base
dvc exp run -f -n 'high_loc' -S 'generate.r_loc=[10.0,13.0]' -S 'generate.r_scale=[0.1,1.0]' -S 'train.dense_dropout=0.5' -S 'train.dense_size=64'
cp metrics/*.json evaluation/*.png results/high_loc
dvc exp run -f -n 'high_scale' -S 'generate.r_loc=[-3.0,3.0]' -S 'generate.r_scale=[1.0,10.0]' -S 'train.dense_dropout=0.5' -S 'train.dense_size=64'
cp metrics/*.json evaluation/*.png results/high_scale
dvc exp run -f -n 'high_dropout' -S 'generate.r_loc=[-3.0,3.0]' -S 'generate.r_scale=[0.1,1.0]' -S 'train.dense_dropout=0.9' -S 'train.dense_size=64'
cp metrics/*.json evaluation/*.png results/high_dropout
dvc exp run -f -n 'low_dropout' -S 'generate.r_loc=[-3.0,3.0]' -S 'generate.r_scale=[0.1,1.0]' -S 'train.dense_dropout=0.1' -S 'train.dense_size=64'
cp metrics/*.json evaluation/*.png results/low_dropout
dvc exp run -f -n 'small_size' -S 'generate.r_loc=[-3.0,3.0]' -S 'generate.r_scale=[0.1,1.0]' -S 'train.dense_dropout=0.5' -S 'train.dense_size=16'
cp metrics/*.json evaluation/*.png results/small_size
dvc exp run -f -n 'large_size' -S 'generate.r_loc=[-3.0,3.0]' -S 'generate.r_scale=[0.1,1.0]' -S 'train.dense_dropout=0.5' -S 'train.dense_size=1024'
cp metrics/*.json evaluation/*.png results/large_size