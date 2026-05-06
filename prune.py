import torch
import torch.nn as nn

def get_prunable_layers(model):
    """Return a list of (name, module) for layers that should be pruned."""
    return [(name, module) for name, module in model.named_modules() if isinstance(module, (nn.Conv2d, nn.Linear))]

def compute_importance_gradients(model, train_loader, criterion, num_batches=3, device="cpu"):
    """
    Computes gradients on a few batches to be used for importance calculation.
    We accumulate the gradients, so zero_grad is called once at the start.
    """
    model.train()
    model.zero_grad()
    batches_processed = 0
    for data, target in train_loader:
        if batches_processed >= num_batches:
            break
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        batches_processed += 1

def compute_masks(model, current_masks, prune_percent=0.10, method="gradient"):
    """
    Calculates layer-wise importance and computes new binary masks.
    Supported methods: "gradient" (|weight x gradient|) and "random".
    """
    new_masks = {}
    total_weights = 0
    total_pruned = 0
    
    for name, module in get_prunable_layers(model):
        weight = module.weight.data
        
        if method == "gradient":
            if module.weight.grad is None:
                importance = torch.ones_like(weight)
            else:
                grad = module.weight.grad.data
                importance = torch.abs(weight * grad)
        elif method == "random":
            importance = torch.rand_like(weight)
        else:
            raise ValueError(f"Unknown pruning method: {method}")
        
        mask = current_masks.get(name, torch.ones_like(weight))
        
        num_total_elements = weight.numel()
        num_to_prune_this_iter = int(num_total_elements * prune_percent)
        num_already_pruned = int((mask == 0).sum().item())
        target_pruned = num_already_pruned + num_to_prune_this_iter
        
        if target_pruned >= num_total_elements:
            target_pruned = num_total_elements - 1
            
        masked_importance = importance.clone()
        # Force already pruned weights to have the lowest importance so they stay pruned
        masked_importance[mask == 0] = -1.0 
        
        flat_importance = masked_importance.view(-1)
        sorted_indices = torch.argsort(flat_importance)
        
        new_mask = torch.ones_like(flat_importance)
        if target_pruned > 0:
            new_mask[sorted_indices[:target_pruned]] = 0.0
        new_mask = new_mask.view_as(weight)
        
        new_masks[name] = new_mask
        
        total_weights += num_total_elements
        total_pruned += target_pruned
        
    return new_masks, total_pruned, total_weights

def apply_pruning_mask(model, masks):
    """Applies the mask by zeroing out the weights that are pruned."""
    with torch.no_grad():
        for name, module in model.named_modules():
            if name in masks:
                module.weight.data.mul_(masks[name])

def enforce_mask_gradients(model, masks):
    """Zeros out the gradients for pruned weights to prevent them from updating."""
    for name, module in model.named_modules():
        if name in masks and module.weight.grad is not None:
            module.weight.grad.data.mul_(masks[name])
