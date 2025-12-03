import torch
import random
from typing import Dict, Any, List

def get_device(device_index=0):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_index}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def prediction_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns per-sample entropy over the softmax distribution.

    Args:
        logits: A tensor of shape (batch_size, num_classes) containing the logits.

    Returns:
        A tensor of shape (batch_size,) containing the entropy for each sample.
    """
    p = torch.nn.Softmax(dim=1)(logits) + 1e-8
    return (-p * p.log()).sum(1)

def strip_scores(model, dataloader, num_bases: int = 32, num_perturbations: int = 16, device=None) -> Dict[str, Any]:
    """
    Computes STRIP-style entropy scores.

    Args:
        model: The model to evaluate.
        dataloader: Dataloader providing the data.
        num_bases: Number of base samples to evaluate.
        num_perturbations: Number of perturbations per base sample.
        device: Device to run the computation on.

    Returns:
        A dictionary containing the raw entropy scores.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = get_device(0)

    model.to(device, dtype=torch.float32)
    model.eval()

    # Collect all images from the dataloader to use as a pool for mixing
    all_images = []
    for images, _ in dataloader:
        all_images.append(images)
        if len(all_images) * images.shape[0] >= num_bases + num_perturbations * 2: # Heuristic to stop early if we have enough data
             break
    
    if not all_images:
         raise ValueError("Dataloader is empty")

    all_images = torch.cat(all_images, dim=0)
    
    # Ensure we have enough images
    if len(all_images) < num_bases:
        num_bases = len(all_images)
        # raise ValueError(f"Not enough images in dataloader. Needed {num_bases}, got {len(all_images)}")

    # Select base samples
    indices = torch.randperm(len(all_images))
    base_indices = indices[:num_bases]
    base_images = all_images[base_indices].to(device, dtype=torch.float32)

    import numpy as np
    entropies_list = []

    with torch.no_grad():
        for i in range(num_bases):
            base_img = base_images[i]
            perturb_indices = torch.randint(0, len(all_images), (num_perturbations,))
            perturb_images = all_images[perturb_indices].to(device, dtype=torch.float32)
            mixed_images = 0.5 * base_img.unsqueeze(0) + 0.5 * perturb_images
            logits = model(mixed_images)
            entropies = prediction_entropy(logits)
            mean_entropy = entropies.mean().item()
            entropies_list.append(mean_entropy)

    # Compute summary statistics and verdict
    mean_entropy = float(np.mean(entropies_list)) if entropies_list else 0.0
    # Example threshold for STRIP (tune as needed)
    threshold = 1
    verdict = "attack" if mean_entropy < threshold else "clean"
    suspected_target = None  # STRIP does not identify a target class

    # Try to get dataset name
    dataset_name = getattr(getattr(dataloader, 'dataset', None), '__class__', None)
    dataset_name = dataset_name.__name__.lower() if dataset_name else "unknown"

    return {
        "defense": "strip",
        "entropies": entropies_list,
        "mean_entropy": mean_entropy,
        "verdict": verdict,
        "suspected_target": suspected_target,
        "thresholds": {
            "mean_entropy": threshold
        },
        "num_bases": num_bases,
        "num_perturbations": num_perturbations,
        "parameters": {
            "device": str(device),
        },
        "dataset": dataset_name
    }
