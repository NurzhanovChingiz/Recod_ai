from tqdm import tqdm
from config import CFG
from pathlib import Path
from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score
from utils.metrics.F1 import calculate_f1_score as f1_score
import os
from PIL import Image

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
from torch.utils.data import DataLoader
from utils.dataset.ForgerySegDataset import ForgerySegDataset

from utils.train_func import train
from utils.validate_func import validate_model
from inference import pipeline_final
import numpy as np
import torch
def main():
    auth_imgs = sorted([str(Path(CFG.AUTH_DIR)/f) for f in os.listdir(CFG.AUTH_DIR)])
    forg_imgs = sorted([str(Path(CFG.FORG_DIR)/f) for f in os.listdir(CFG.FORG_DIR)])
    
    train_auth, val_auth = train_test_split(auth_imgs, test_size=CFG.TEST_SIZE, random_state=CFG.SEED)
    train_forg, val_forg = train_test_split(forg_imgs, test_size=CFG.TEST_SIZE, random_state=CFG.SEED)
    print(f"Train authentic: {len(train_auth)}, Train forged: {len(train_forg)}")
    print(f"Val authentic: {len(val_auth)}, Val forged: {len(val_forg)}")
    
    train_dataset = ForgerySegDataset(train_auth, train_forg, CFG.MASK_DIR, transform=CFG.TRAIN_TRANSFORM)
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_dataset = ForgerySegDataset(val_auth, val_forg, CFG.MASK_DIR, transform=CFG.TEST_TRANSFORM)
    test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    for epoch in range(CFG.EPOCHS):
        train(
            CFG.MODEL,
            train_loader,
            CFG.LOSS_FN,
            CFG.OPTIMIZER,
            CFG.DEVICE,
            epoch,
            scaler=CFG.SCALER
            )
        validate_model(CFG.MODEL, test_loader, CFG.DEVICE, epoch, scaler=CFG.SCALER)
    checkpoint = {
        'model': CFG.MODEL.state_dict(),
        'optimizer': CFG.OPTIMIZER.state_dict(),
        'scaler': CFG.SCALER.state_dict() if CFG.SCALER is not None else None
    }
    if CFG.SAVE_MODEL:
        torch.save(checkpoint, CFG.PATH_SAVE_MODEL)
    
    
    #  VALIDATION/SCORING 
    print("\n--- Validation/Scoring ---")
    
    # Evaluate on a sample of forged images
    val_items = [(p, 1) for p in val_forg[:50]]
    results = []
    
    for p, _ in tqdm(val_items, desc="Validation forged-only"):
        try:
            pil = Image.open(p).convert("RGB")
            
            # Get prediction
            label, m_pred, dbg = pipeline_final(pil, CFG.MODEL)
            
            # Load ground truth mask
            m_gt = np.load(Path(CFG.MASK_DIR)/f"{Path(p).stem}.npy")
            if m_gt.ndim == 3: m_gt = np.max(m_gt, axis=0)
            m_gt = (m_gt > 0).astype(np.uint8)
            
            # Ensure predicted mask is a binary numpy array (zero mask if filtered)
            m_pred = (m_pred > 0).astype(np.uint8) if m_pred is not None else np.zeros_like(m_gt)
            
            # Calculate F1-score for segmentation
            f1 = f1_score(m_gt.flatten(), m_pred.flatten())
            results.append((Path(p).stem, f1, dbg))
            
        except Exception as e:
            print(f"Error processing {p}: {e}")
            
    print("\nF1-score per forged image (Segmentation):")
    for cid, f1, dbg in results:
        print(f"{cid} — F1={f1:.4f} | area={dbg['area']} mean={dbg['mean_inside']:.3f} thr={dbg['thr']:.3f}")
        
    mean_f1 = np.mean([r[1] for r in results])
    print(f"\nAverage F1 (Forged Segmentation) = {mean_f1:.4f}")
if __name__ == "__main__":
    main()
    
    
    # avg_loss=0.7419 05:00 Average F1 (Forged Segmentation) = 0.0274
    # batch 16 02:55