import argparse
import sys
import platform
import time
import traceback

def main():
    parser = argparse.ArgumentParser(description="MNIST CNN Training")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("MNIST CNN Training - Verbose Mode")
    print("=" * 60)

    # --- Environment ---
    print("\n[ENV] Python version:", sys.version)
    print("[ENV] Platform:", platform.platform())

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        from torchvision import datasets, transforms
        print(f"[ENV] PyTorch version: {torch.__version__}")
        try:
            import torchvision
            print(f"[ENV] TorchVision version: {torchvision.__version__}")
        except Exception:
            pass
    except ImportError as e:
        print(f"[ERROR] Failed to import PyTorch/TorchVision: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- GPU Detection ---
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

    torch.manual_seed(args.seed)

    # --- Data Loading ---
    print("\n[DATA] Loading MNIST dataset...")
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
        print(f"[DATA] Train samples: {len(train_dataset)}")
        print(f"[DATA] Test samples: {len(test_dataset)}")
        print(f"[DATA] Train batches: {len(train_loader)}")
        print(f"[DATA] Batch size: 64 (train) / 1000 (test)")
    except Exception as e:
        print(f"[ERROR] Failed to load MNIST dataset: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Model Definition ---
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc1 = nn.Linear(32 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(-1, 32 * 7 * 7)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    print("\n[MODEL] Creating SimpleCNN...")
    try:
        model = SimpleCNN().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[MODEL] Total parameters: {total_params:,}")
        print(f"[MODEL] Trainable parameters: {trainable_params:,}")
        print(f"[MODEL] Model device: {next(model.parameters()).device}")
        print(f"[MODEL] Architecture:")
        for name, layer in model.named_children():
            print(f"[MODEL]   {name}: {layer}")
    except Exception as e:
        print(f"[ERROR] Model creation failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    print(f"[TRAIN] Optimizer: SGD(lr=0.01)")

    # --- Training ---
    print(f"\n[TRAIN] Starting training for {args.epochs} epochs...")
    total_start = time.time()
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_start = time.time()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                if (batch_idx + 1) % 200 == 0:
                    print(f"[TRAIN]   Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

            avg_loss = epoch_loss / len(train_loader)
            epoch_time = time.time() - epoch_start
            print(f"[TRAIN] Epoch {epoch}/{args.epochs} | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

        total_time = time.time() - total_start
        print(f"\n[TRAIN] Training completed in {total_time:.2f}s")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Evaluation ---
    print("\n[EVAL] Evaluating model on test set...")
    try:
        model.eval()
        correct = 0
        total = 0
        class_correct = [0] * 10
        class_total = [0] * 10
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                for i in range(target.size(0)):
                    label = target[i].item()
                    class_total[label] += 1
                    if pred[i].item() == label:
                        class_correct[label] += 1

        accuracy = 100.0 * correct / total
        print(f"[EVAL] Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
        print(f"\n[EVAL] Per-class accuracy:")
        for cls in range(10):
            if class_total[cls] > 0:
                cls_acc = 100.0 * class_correct[cls] / class_total[cls]
                print(f"[EVAL]   Digit {cls}: {cls_acc:.1f}% ({class_correct[cls]}/{class_total[cls]})")
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\n[DONE] Experiment completed successfully (seed={args.seed}, epochs={args.epochs})")

if __name__ == "__main__":
    main()
