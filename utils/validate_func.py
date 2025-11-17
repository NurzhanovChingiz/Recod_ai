import torch
import numpy as np
from tqdm import tqdm
from utils.metrics.F1 import calculate_f1_score as f1_score

def validate_model(model, dataloader, device, epoch, scaler=None):
    from config import CFG
    model.eval()
    total_f1 = 0
    count = 0
    with torch.no_grad():
        for x, m in tqdm(dataloader, desc=f"[Segmentation] Epoch {epoch+1}/{CFG.EPOCHS}"):
            x, m = x.to(device), m.to(device)
            if CFG.USE_AMP and scaler is not None:
                with torch.amp.autocast(device_type=device.type, dtype=CFG.SCALER_DTYPE, enabled=True):
                    preds = model.forward_seg(x, (CFG.IMG_SIZE, CFG.IMG_SIZE))

                    preds_np = (preds.sigmoid().detach().cpu().numpy() > 0.5).astype(np.uint8)
                    m_np = m.detach().cpu().numpy().astype(np.uint8)

                    for pred_mask, gt_mask in zip(preds_np, m_np):
                        f1 = f1_score(pred_mask.flatten(), gt_mask.flatten())
                        total_f1 += f1
                        count += 1
            else:
                preds = model.forward_seg(x, (CFG.IMG_SIZE, CFG.IMG_SIZE))
                
                preds_np = (preds.sigmoid().detach().cpu().numpy() > 0.5).astype(np.uint8)
                m_np = m.detach().cpu().numpy().astype(np.uint8)
                
                for pred_mask, gt_mask in zip(preds_np, m_np):
                    f1 = f1_score(pred_mask.flatten(), gt_mask.flatten())
                    total_f1 += f1
                    count += 1
    avg_f1 = total_f1 / count if count > 0 else 0
    print(f"\n--- Validation/Scoring ---\nAverage F1 Score: {avg_f1:.4f}")
    
    # from config import CFG
    # model.eval()
    # total_f1 = 0
    # count = 0
    # with torch.no_grad():
    #     for x, m in tqdm(dataloader, desc=f"[Segmentation] Epoch {epoch+1}/{CFG.EPOCHS}"):
    #         x, m = x.to(device), m.to(device)
            
    #         if scaler is not None:
    #             with torch.amp.autocast(device_type=device.type):
    #                 preds = model.forward_seg(x, (CFG.IMG_SIZE, CFG.IMG_SIZE))
    #         else:
    #             preds = model.forward_seg(x, (CFG.IMG_SIZE, CFG.IMG_SIZE))
                
    #         preds_np = (preds.sigmoid().cpu().numpy() > 0.5).astype(np.uint8)
    #         m_np = m.cpu().numpy().astype(np.uint8)
            
    #         for pred_mask, gt_mask in zip(preds_np, m_np):
    #             f1 = f1_score(pred_mask.flatten(), gt_mask.flatten())
    #             total_f1 += f1
    #             count += 1
    # avg_f1 = total_f1 / count if count > 0 else 0
    # print(f"\n--- Validation/Scoring ---\nAverage F1 Score: {avg_f1:.4f}")