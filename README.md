# PyTorch Simple Training

Two minimal PyTorch training scripts.

## Scripts

### train_linear.py

Linear regression on synthetic data (y = 3x + 2 + noise).

```bash
python train_linear.py --seed 42 --epochs 50
```

### train_mnist.py

Simple CNN trained on MNIST.

```bash
python train_mnist.py --seed 42 --epochs 3
```

## Setup

```bash
pip install -r requirements.txt
```
