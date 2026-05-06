import torch
from prune import enforce_mask_gradients, apply_pruning_mask

def train_epoch(model, train_loader, optimizer, criterion, masks=None, device="cpu"):
    """
    Trains the model for one epoch.
    If masks are provided, enforces sparsity on weights and gradients.
    """
    model.train()
    running_loss = 0.0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Zero out gradients for pruned weights before step
        if masks is not None:
            enforce_mask_gradients(model, masks)
            
        optimizer.step()
        
        # Ensure weights remain exactly zero (e.g. from momentum updates)
        if masks is not None:
            apply_pruning_mask(model, masks)
            
        running_loss += loss.item() * data.size(0)
        
    return running_loss / len(train_loader.dataset)
