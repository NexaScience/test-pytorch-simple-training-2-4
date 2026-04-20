import argparse
import sys
import platform
import time
import traceback

def main():
    parser = argparse.ArgumentParser(description="Linear Regression Training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Linear Regression Training - Verbose Mode")
    print("=" * 60)

    # --- Environment Info ---
    print("\n[ENV] Python version:", sys.version)
    print("[ENV] Platform:", platform.platform())

    # --- Import & GPU Detection ---
    try:
        import torch
        import torch.nn as nn
        print(f"\n[ENV] PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"[ERROR] Failed to import PyTorch: {e}")
        traceback.print_exc()
        sys.exit(1)

    # GPU Detection
    print("\n[DEVICE] Checking for GPU availability...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[DEVICE] CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"[DEVICE] CUDA version: {torch.version.cuda}")
        print(f"[DEVICE] GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[DEVICE] Apple MPS available")
    else:
        device = torch.device("cpu")
        print("[DEVICE] No GPU available - using CPU")
    print(f"[DEVICE] Selected device: {device}")

    # --- Data Generation ---
    print(f"\n[DATA] Generating synthetic data (y = 3x + 2 + noise, seed={args.seed})...")
    try:
        torch.manual_seed(args.seed)
        X = torch.randn(200, 1)
        y = 3 * X + 2 + 0.5 * torch.randn(200, 1)
        X, y = X.to(device), y.to(device)
        print(f"[DATA] Samples: {X.shape[0]}, Features: {X.shape[1]}")
        print(f"[DATA] X range: [{X.min().item():.4f}, {X.max().item():.4f}]")
        print(f"[DATA] y range: [{y.min().item():.4f}, {y.max().item():.4f}]")
        print(f"[DATA] Data device: {X.device}")
    except Exception as e:
        print(f"[ERROR] Data generation failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Model Setup ---
    print("\n[MODEL] Creating Linear(1, 1) model...")
    try:
        model = nn.Linear(1, 1).to(device)
        print(f"[MODEL] Parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"[MODEL] Model device: {next(model.parameters()).device}")
        print(f"[MODEL] Initial weight: {model.weight.item():.4f}, bias: {model.bias.item():.4f}")
    except Exception as e:
        print(f"[ERROR] Model creation failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    lr = 0.05
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    print(f"[TRAIN] Optimizer: SGD(lr={lr})")
    print(f"[TRAIN] Loss function: MSELoss")

    # --- Training Loop ---
    print(f"\n[TRAIN] Starting training for {args.epochs} epochs...")
    start_time = time.time()
    try:
        for epoch in range(1, args.epochs + 1):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch == 1:
                w = model.weight.item()
                b = model.bias.item()
                print(f"[TRAIN] Epoch {epoch:3d}/{args.epochs} | loss={loss.item():.4f} | weight={w:.4f} | bias={b:.4f}")

        train_time = time.time() - start_time
        print(f"\n[TRAIN] Training completed in {train_time:.4f}s")
    except Exception as e:
        print(f"[ERROR] Training failed at epoch {epoch}: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Results ---
    final_w = model.weight.item()
    final_b = model.bias.item()
    final_loss = loss.item()
    print(f"\n[RESULT] Final loss: {final_loss:.4f}")
    print(f"[RESULT] Learned weight: {final_w:.4f} (true: 3.0, error: {abs(final_w - 3.0):.4f})")
    print(f"[RESULT] Learned bias:   {final_b:.4f} (true: 2.0, error: {abs(final_b - 2.0):.4f})")
    print(f"\n[DONE] Experiment completed successfully (seed={args.seed}, epochs={args.epochs})")

if __name__ == "__main__":
    main()
