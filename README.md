# DS3L
This is the code for paper "Safe Deep Semi-Supervised Learning for Unseen-Class Unlabeled Data" published in ICML 2020.

# Setups

The code is implemented with Python and Pytorch.

# Running D3SL for benchmark datasets

Here is an example:

```bash
python train.py --dataset MNIST --ratio 0.6 --n_labels 60 --iterations 200000
```

# Acknowledgements
We thank the Pytorch implementation on Meta-Net (https://github.com/xjtushujun/meta-weight-ne) and learning-to-reweight-examples(https://github.com/danieltan07/learning-to-reweight-examples).


# Contact 
If you have any questions, please contact Lan-Zhe Guo (guolz@lamda.nju.edu.cn).
