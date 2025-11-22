
from tqdm import tqdm
from config import CFG
import torch



def train(model, dataloader, loss_fn, optimizer, device, epoch, scaler=None):
    model.train()
    total_loss = 0
    batch_count = 0
    nan_count = 0
    for batch_idx, (x, m) in enumerate(tqdm(dataloader, desc=f"[Segmentation] Epoch {epoch+1}/{CFG.EPOCHS}")):
        
        x, m = x.to(device), m.to(device)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: Invalid input image values detected in batch {batch_idx}")
            nan_count += 1
            continue
        if torch.isnan(m).any() or torch.isinf(m).any():
            print(f"Warning: Invalid mask values detected in batch {batch_idx}")
            nan_count += 1
            continue
        # Backward pass with conditional scaler
        optimizer.zero_grad(set_to_none=True)
        # Forward pass with conditional autocast
        if CFG.USE_AMP:
            with torch.amp.autocast(device_type=device.type, dtype=CFG.SCALER_DTYPE, enabled=True):
                pred = model.forward_seg(x, (CFG.IMG_SIZE, CFG.IMG_SIZE))
                loss = loss_fn(pred, m)
        else:
            pred = model.forward_seg(x, (CFG.IMG_SIZE, CFG.IMG_SIZE))
            loss = loss_fn(pred, m)
        if CFG.USE_AMP:
            # AMP training
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # update generator weights
            scaler.step(optimizer)
            
            scaler.update()
        else:
            # Regular training
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
            batch_count += 1
        
    if batch_count > 0:
        avg_loss = total_loss / batch_count
    else:
        avg_loss = float('nan')
        print("Warning: No valid batches processed!")
    if nan_count > 0:
        print(f"Skipped {nan_count} batches due to NaN/Inf values.")
    print(f"Avg_loss={avg_loss:.4f}, Valid batches: {batch_count}/{len(dataloader)}")
    
    
    #     # Check for NaN loss
    #     if torch.isnan(loss) or torch.isinf(loss):
    #         print(f"Warning: Invalid loss detected, skipping batch")
    #         continue
    #     if scaler is None:
    #         loss.backward()
    #         optimizer.step()
    #     else:
    #         scaler.scale(loss).backward()
        
    #     # Clip gradients to prevent exploding gradients
    #     if scaler is None:
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #     else:
    #         scaler.unscale_(optimizer)
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
    #     if scaler is not None:
    #         scaler.step(optimizer)
    #         scaler.update()
    #     total_loss += loss.item()
    # avg_loss = total_loss / len(dataloader)
    # print(f"  → avg_loss={avg_loss:.4f}")
    
    
