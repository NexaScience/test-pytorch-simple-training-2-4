# PyTorch Simple Training

Minimal PyTorch training scripts for testing and demonstration.

## Scripts

### train_mnist.py

Trains a simple 2-layer CNN on the MNIST handwritten digit dataset.

```bash
python train_mnist.py --epochs 5 --batch-size 64 --lr 0.01
```

Options:
- `--epochs` - Number of training epochs (default: 5)
- `--batch-size` - Training batch size (default: 64)
- `--lr` - Learning rate (default: 0.01)
- `--seed` - Random seed (default: 42)
- `--no-cuda` - Disable CUDA training

### train_linear.py

Trains a linear regression model on synthetic data (y = 3x + 2 + noise).

```bash
python train_linear.py --epochs 100 --n-samples 1000 --lr 0.01
```

Options:
- `--n-samples` - Number of data samples (default: 1000)
- `--epochs` - Number of training epochs (default: 100)
- `--lr` - Learning rate (default: 0.01)
- `--seed` - Random seed (default: 42)

## Setup

```bash
pip install -r requirements.txt
```
