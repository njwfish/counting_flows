"""
Clean Training Functions for Flow-based Models with Modular Architecture

Simplified training loop that works with various bridges (CFM, counting flows, etc.) and loss functions.
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
import random

from training import Trainer


def sparse_aggregation_collate_fn(batch: List[Dict[str, Any]], max_batch_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Flexible custom collate function that handles nested structures and variable group sizes.
    
    This function:
    1. Detects which tensors have group_size dimension (same shape[0] as x_1)
    2. Flattens group_size tensors while preserving nested structure  
    3. Stacks non-group_size tensors normally along batch dimension
    4. Creates sparse aggregation matrix for reconstruction
    
    Args:
        batch: List of dataset items with arbitrary nested structure.
               Must contain 'x_1' tensor to determine group_size.
    
    Returns:
        Collated batch with same structure as input, where:
        - Group-size tensors are flattened: [total_samples, ...other_dims]
        - Non-group-size tensors are stacked: [batch_size, ...other_dims]  
        - Added keys: A_sparse, group_sizes
    """
    if not batch:
        return {}
        
    batch_size = len(batch)
    
    # Extract group sizes from x_1 (which must exist)
    # Handle case where x_1 might be nested dict
    def _get_group_size(x_1):
        if isinstance(x_1, dict):
            # If x_1 is a dict, use the first tensor we find
            for key, value in x_1.items():
                if isinstance(value, torch.Tensor):
                    return value.shape[0]
            raise ValueError("No tensor found in x_1 dict")
        else:
            return x_1.shape[0]

    def _filter_batch_by_size(batch, group_sizes, max_batch_size):
        """Filter batch to respect max_batch_size constraints"""
        # Filter oversized groups with warning
        valid_items = [(item, size) for item, size in zip(batch, group_sizes) if size <= max_batch_size]
        if len(valid_items) < len(batch):
            dropped = len(batch) - len(valid_items)
            print(f"Warning: Dropped {dropped} group(s) exceeding max_batch_size={max_batch_size}")
        
        if not valid_items:
            return [], []
        
        items, sizes = zip(*valid_items)
        
        # Random dropping until under threshold
        indices = list(range(len(items)))
        while sum(sizes[i] for i in indices) > max_batch_size:
            indices.pop(random.randint(0, len(indices) - 1))
        
        return len(indices), [items[i] for i in indices], [sizes[i] for i in indices]
    
    group_sizes = [_get_group_size(item['x_1']) for item in batch]
    if max_batch_size is not None:
        batch_size, batch, group_sizes = _filter_batch_by_size(batch, group_sizes, max_batch_size)

    total_samples = sum(group_sizes)
    
    def _is_group_tensor(tensor, group_size):
        """Check if tensor has group_size as first dimension"""
        return isinstance(tensor, torch.Tensor) and len(tensor.shape) > 0 and tensor.shape[0] == group_size
    
    def _allocate_tensor(sample_tensor, is_group_tensor, total_size, batch_size):
        """Allocate tensor for collation"""
        if is_group_tensor:
            # Group tensor: flatten first dimension
            new_shape = (total_size,) + sample_tensor.shape[1:]
        else:
            # Non-group tensor: add batch dimension
            new_shape = (batch_size,) + sample_tensor.shape
        return torch.zeros(new_shape, dtype=sample_tensor.dtype)
    
    def _collate_nested_dict(batch_dicts, path=""):
        """Recursively collate nested dictionary structures"""
        if not batch_dicts:
            return {}
            
        result = {}
        sample_dict = batch_dicts[0]
        
        for key in sample_dict.keys():
            values = [item[key] for item in batch_dicts]
            
            if isinstance(values[0], dict):
                # Recursively handle nested dicts
                result[key] = _collate_nested_dict(values, f"{path}.{key}")
            elif isinstance(values[0], torch.Tensor):
                # Handle tensor collation
                sample_tensor = values[0]
                # Check if this tensor has group_size dimension by comparing with each batch item
                # A tensor is a group tensor if ALL batch items have matching first dimension to their group_size
                # AND the tensor has more than 1 dimension (to avoid confusion with feature vectors)
                is_group_tensor = (
                    len(sample_tensor.shape) > 1 and  # Must be at least 2D
                    all(_is_group_tensor(values[i], group_sizes[i]) for i in range(len(values)))
                )
                
                if is_group_tensor:
                    # Flatten group tensors
                    collated_tensor = _allocate_tensor(sample_tensor, True, total_samples, batch_size)
                    current_offset = 0
                    
                    for batch_idx, tensor in enumerate(values):
                        group_size = group_sizes[batch_idx]
                        end_offset = current_offset + group_size
                        collated_tensor[current_offset:end_offset] = tensor
                        current_offset = end_offset
                        
                    result[key] = collated_tensor
                else:
                    # Stack non-group tensors normally
                    result[key] = torch.stack(values)
            else:
                # Handle non-tensor values (lists, scalars, etc.)
                # For now, just keep as list - could be made more sophisticated
                result[key] = values
                
        return result
    
    # Collate the nested structure
    result = _collate_nested_dict(batch)
    
    # Create sparse aggregation matrix
    # A[i, j] = 1 if sample j belongs to group i, 0 otherwise
    row_indices = []
    col_indices = []
    
    current_offset = 0
    for batch_idx, group_size in enumerate(group_sizes):
        for sample_idx in range(current_offset, current_offset + group_size):
            row_indices.append(batch_idx)
            col_indices.append(sample_idx)
        current_offset += group_size
    
    # Create sparse aggregation matrix
    values = torch.ones(len(row_indices), dtype=torch.float32)
    indices = torch.stack([torch.tensor(row_indices), torch.tensor(col_indices)])
    A_sparse = torch.sparse_coo_tensor(
        indices, values, (batch_size, total_samples), dtype=torch.float32
    ).coalesce()
    
    # Add aggregation metadata
    result['A'] = A_sparse.to_dense() # if A_sparse.shape[0] == 1 else A_sparse
    result['group_sizes'] = torch.tensor(group_sizes, dtype=torch.long)
    return result

def deconv_sample_batch(
    batch: Dict[str, Any], 
    device: str="cuda", 
    condition_on_end_time: bool=False
) -> Dict[str, Any]:
    if isinstance(batch['x_1'], dict):
        x_1 = {}
        for k in batch['x_1']:
            x_1[k] = batch['x_1'][k].to(device)
    else:
        x_1 = batch['x_1'].to(device)
    
    if isinstance(batch['X_0'], dict):
        for k in batch['X_0']:
            batch['X_0'][k] = batch['X_0'][k].to(device)
    else:
        batch['X_0'] = batch['X_0'].to(device)
    
    context = {'target_sum': batch['X_0']}
    for key in batch:
        if key not in ['x_1', 'x_0', 'X_0', 'group_sizes']:
            batch[key] = batch[key].to(device)
            context[key] = batch[key].to(device)

    if isinstance(batch['x_0'], dict):
        # this lets us condition on x_0 if it is provided (and not in X_0)
        # using the fact that the model "samples" from x_0 if x_0 is provided, e.g. "conditional" sampling
        # across modalities 
        context_x_0 = {}
        for k in batch['x_0']:
            if k not in batch['X_0']:
                context_x_0[k] = batch['x_0'][k].to(device)

        context['x_0'] = context_x_0

    sampler_kwargs = {
            'start_times': {k: 0.0 for k in context['x_0']},
            'x_start_time': {k: context['x_0'][k] for k in context['x_0']}
    } if condition_on_end_time else {}

    return x_1, context, sampler_kwargs


class DeconvTrainer(Trainer):
    """
    Clean, modular trainer for flow-based models.
    Works with any bridge (CFM, counting flows, etc.) and loss function (energy score, MSE, etc.).
    """
    
    def __init__(
        self,
        condition_on_end_time: bool = False,
        max_batch_size: Optional[int] = None,
        direct_only: bool = False,
        **kwargs
    ):
        """
        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            num_epochs: Number of training epochs
            device: Device for model computation
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle dataset
            num_workers: Number of DataLoader workers
            print_every: Print loss every N epochs
            save_every: Save checkpoint every N epochs (None to disable)
            output_dir: Output directory for checkpoints
        """
        super().__init__(
            **kwargs
        )
        self.condition_on_end_time = condition_on_end_time
        self.max_batch_size = max_batch_size
        self.direct_only = direct_only

    def _create_dataloader(self, dataset):
        """Create DataLoader from dataset with custom sparse aggregation collate function"""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=lambda x: sparse_aggregation_collate_fn(x, self.max_batch_size)
        )

    def training_step(self, model: torch.nn.Module, bridge: Any, batch: Dict[str, torch.Tensor], avg_model: Optional[torch.optim.swa_utils.AveragedModel] = None) -> float:
        """Execute a single training step"""
        # Extract data from batch
        if isinstance(batch['x_0'], dict):
            x_0, x_1 = {}, {}
            for k in batch['x_0']:
                x_0[k] = batch['x_0'][k].to(self.device)
                x_1[k] = batch['x_1'][k].to(self.device)
        else:
            x_0 = batch['x_0'].to(self.device)  # Target counts
            x_1 = batch['x_1'].to(self.device)  # Source counts  

        A = batch['A']
        
        # Apply bridge to get diffusion samples
        if self.direct_only:
            t, x_t, target = torch.ones(x_0.shape[0], 1).float().cuda(), x_1.float().cuda(), x_0.float().cuda()
        else:
            t, x_t, target = bridge(x_0, x_1)
        
        inputs = {
            "t": t,
            "x_t": x_t,
            "A": A,
        }
        # target = batch['X_0'] - A @ x_t if bridge.delta else batch['X_0']
        if isinstance(x_0, dict):
            target = {}
            target['x_0'] = x_0.to(self.device)
            target['X_0'] = batch['X_0'].to(self.device)
        else:
            target = batch['X_0'].to(self.device)

        for key in batch:
            if key != 'x_0' and key != 'x_1' and key != 'A' and key != 'group_sizes' and key != 'X_0':
                inputs[key] = batch[key].to(self.device)
                if 'key' == 'class_emb' and self.classifier_free_guidance_prob > 0:
                    # random mask out prob of the class embeddings
                    mask = torch.rand(inputs[key].shape[0]) < self.classifier_free_guidance_prob
                    inputs[key][mask] = 0
        
        # Training step
        # print(A.shape)
        self.optimizer.zero_grad()
        loss = model.loss(target, inputs, agg=A)
        loss.backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad_norm)
        self.optimizer.step()

        if avg_model is not None:
            avg_model.update_parameters(model)

        if self.scheduler is not None:
            self.scheduler.step()
        
        return loss.item()
    
    def train_epoch(self, epoch: int, model: torch.nn.Module, bridge: Any, dataloader: DataLoader, avg_model: Optional[torch.optim.swa_utils.AveragedModel] = None) -> float:
        """Train for one epoch"""
        epoch_losses = []

        from tqdm import tqdm
        # e step
        if len(dataloader) > 20:
            tqdm = tqdm(dataloader)
        else:
            tqdm = dataloader
        for batch in tqdm:
            # extract x_1 in multimodal and single modality cases
            if not self.direct_only:
                model.eval()
                avg_model.eval()
                with torch.no_grad():
                    x_1, context, sampler_kwargs = deconv_sample_batch(batch, self.device, self.condition_on_end_time)

                    # if multidimensional time and we have x_0 we should condition on the start time zero for the observed modalities
                    batch['x_0'] = bridge.sampler(
                        x_1, context, avg_model.module.to(self.device) if avg_model is not None else model, 
                        **sampler_kwargs
                    )
            else:
                batch['A'] = batch['A'].to(self.device)

            # m step
            model.train()
            avg_model.train()
            # sample time step
            batch_loss = self.training_step(model, bridge, batch, avg_model=avg_model)
            epoch_losses.append(batch_loss)
            self.losses.append(batch_loss)
        
        return sum(epoch_losses) / len(epoch_losses)
    