#!/usr/bin/env python3
"""Train a linear regression model on synthetic data (y = 3x + 2 + noise)."""

import argparse
import torch
import torch.nn as nn

def main():
    parser = argparse.ArgumentParser(description="Linear regression on synthetic data")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=50, help="training epochs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Generate synthetic data: y = 3x + 2 + noise
    x = torch.randn(200, 1)
    y = 3.0 * x + 2.0 + 0.5 * torch.randn(200, 1)

    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{args.epochs}  loss={loss.item():.4f}")

    print(f"Final loss: {loss.item():.4f}")
    print(f"Learned: weight={model.weight.item():.2f}, bias={model.bias.item():.2f}")

if __name__ == "__main__":
    main()
