import torch
def collate_fn(batch):
    # Transpose the batch (list of lists) to group elements by position
    batch_transposed = list(zip(*batch))
    
    # Process each position across the batch
    processed_batch = []
    for tensors in batch_transposed:
        if all(t is None for t in tensors):  # If all are None, keep None
            processed_batch.append(None)
        else:  # Otherwise, stack non-None tensors and replace None with zero tensors
            #make sure no None element in the tensors
            any_none = any(t is None for t in tensors)
            assert not any_none, "None element in a list of tensors"
            stacked = [
                t for t in tensors
            ]
            processed_batch.append(torch.stack(stacked))
    
    return processed_batch

