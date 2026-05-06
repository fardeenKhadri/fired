import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

from model import SimpleCNN
from utils import get_dataloaders, evaluate
from train import train_epoch
from prune import compute_importance_gradients, compute_masks, apply_pruning_mask

def run_pruning_pipeline(baseline_state, method, train_loader, test_loader, criterion, device):
    print(f"\n{'='*50}")
    print(f"Starting Iterative Pruning: {method.upper()}")
    print(f"{'='*50}")
    
    # Reload identically initialized and trained baseline
    model = SimpleCNN().to(device)
    model.load_state_dict(copy.deepcopy(baseline_state))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    prune_iterations = 8
    prune_percent_per_iter = 0.10
    retrain_epochs = 2
    
    # Evaluate baseline
    test_loss, test_acc = evaluate(model, test_loader, criterion, device=device)
    
    # Track metrics in structured format
    history = {
        "iteration": [0],
        "sparsity": [0.0],
        "accuracy": [test_acc],
        "loss": [test_loss]
    }
    
    masks = {}
    
    for iteration in range(1, prune_iterations + 1):
        print(f"\n[{method.upper()} - Iteration {iteration}/{prune_iterations}]")
        
        # 1. Compute gradients for importance (only needed for gradient method)
        if method == "gradient":
            compute_importance_gradients(model, train_loader, criterion, num_batches=3, device=device)
            
        # 2. Calculate importance & get new masks
        masks, total_pruned, total_weights = compute_masks(model, masks, prune_percent=prune_percent_per_iter, method=method)
        
        sparsity = 100.0 * total_pruned / total_weights
        
        # 3. Apply mask before retraining to ensure zeroed weights
        apply_pruning_mask(model, masks)
        
        # 4. Retrain with strict mask enforcement
        print(f"Retraining for {retrain_epochs} epochs...")
        for epoch in range(retrain_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, masks=masks, device=device)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device=device)
            print(f"  Retrain Epoch {epoch+1}/{retrain_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
            
        history["iteration"].append(iteration)
        history["sparsity"].append(sparsity)
        history["accuracy"].append(test_acc)
        history["loss"].append(test_loss)
        
        print(f"End of Iteration {iteration} | Cumulative Pruned: {sparsity:.1f}% | Final Acc: {test_acc:.2f}%")
        
        # Stop if accuracy degrades significantly
        if test_acc < 90.0:
            print(f"\nAccuracy dropped below 90% ({test_acc:.2f}%). Stopping {method} pruning early.")
            break
            
    return history

def main():
    # Ensure fair comparison with same seed
    torch.manual_seed(42)
    
    device = "cpu"
    print(f"Running on {device}...")
    
    batch_size = 64
    initial_epochs = 5
    
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)
    
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print("\n--- Initial Training Phase (Baseline) ---")
    for epoch in range(initial_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, masks=None, device=device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device=device)
        print(f"Epoch {epoch+1}/{initial_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
    baseline_state = copy.deepcopy(model.state_dict())
    
    results = {}
    
    for method in ["gradient", "random"]:
        results[method] = run_pruning_pipeline(
            baseline_state, method, train_loader, test_loader, criterion, device
        )

    print("\n--- Summary ---")
    for method, history in results.items():
        print(f"\n{method.upper()} Method:")
        for s, a in zip(history["sparsity"], history["accuracy"]):
            print(f"  Sparsity: {s:.1f}% -> Accuracy: {a:.2f}%")
        
    # Plotting
    plt.figure(figsize=(10, 6))
    for method, history in results.items():
        plt.plot(history["sparsity"], history["accuracy"], marker='o', linestyle='-', label=method.capitalize())
        
    plt.title("Sparsity vs Accuracy: Gradient vs Random Pruning")
    plt.xlabel("% Weights Pruned")
    plt.ylabel("Test Accuracy (%)")
    plt.axhline(y=90.0, color='r', linestyle='--', label="90% Threshold")
    plt.legend()
    plt.grid(True)
    plt.savefig("sparsity_comparison.png")
    print("\nPlot saved as sparsity_comparison.png")

if __name__ == "__main__":
    main()
