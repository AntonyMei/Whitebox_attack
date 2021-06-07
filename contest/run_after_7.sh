#!/bin/bash
conda activate tensorflow1
python run_attacks.py --attacks attacker7 --output ./tmp --models cifar10-pgd_at,cifar10-wideresnet_trades,cifar10-feature_scatter,cifar10-robust_overfitting,cifar10-rst,cifar10-fast_at,cifar10-at_he,cifar10-pre_training,cifar10-free_at,cifar10-awp,cifar10-hydra,cifar10-label_smoothing
python run_attacks.py --attacks attacker8 --output ./tmp --models cifar10-pgd_at,cifar10-wideresnet_trades,cifar10-feature_scatter,cifar10-robust_overfitting,cifar10-rst,cifar10-fast_at,cifar10-at_he,cifar10-pre_training,cifar10-free_at,cifar10-awp,cifar10-hydra,cifar10-label_smoothing
python run_attacks.py --attacks attacker9 --output ./tmp --models cifar10-pgd_at,cifar10-wideresnet_trades,cifar10-feature_scatter,cifar10-robust_overfitting,cifar10-rst,cifar10-fast_at,cifar10-at_he,cifar10-pre_training,cifar10-free_at,cifar10-awp,cifar10-hydra,cifar10-label_smoothing
python run_attacks.py --attacks attacker10 --output ./tmp --models cifar10-pgd_at,cifar10-wideresnet_trades,cifar10-feature_scatter,cifar10-robust_overfitting,cifar10-rst,cifar10-fast_at,cifar10-at_he,cifar10-pre_training,cifar10-free_at,cifar10-awp,cifar10-hydra,cifar10-label_smoothing
