#!/usr/bin/env python3
"""Train a simple linear regression model on synthetic data."""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def generate_data(n_samples, seed):
    """Generate synthetic linear data: y = 3x + 2 + noise."""
    np.random.seed(seed)
    x = np.random.uniform(-10, 10, (n_samples, 1)).astype(np.float32)
    noise = np.random.normal(0, 1, (n_samples, 1)).astype(np.float32)
    y = 3.0 * x + 2.0 + noise
    return torch.from_numpy(x), torch.from_numpy(y)


def main():
    parser = argparse.ArgumentParser(description="Train linear regression on synthetic data")
    parser.add_argument("--n-samples", type=int, default=1000, help="number of data samples")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    x, y = generate_data(args.n_samples, args.seed)

    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    print(f"Training linear regression for {args.epochs} epochs with {args.n_samples} samples")

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={loss.item():.4f}")

    weight = model.weight.item()
    bias = model.bias.item()
    print(f"\nLearned parameters: weight={weight:.4f}, bias={bias:.4f}")
    print(f"True parameters:    weight=3.0000, bias=2.0000")


if __name__ == "__main__":
    main()
