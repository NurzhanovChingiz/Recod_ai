import torch
import cv2
import numpy as np
from config import CFG
from utils.clear_gpu import clear_memory

@torch.no_grad()
def segment_prob_map(pil, model):
    """Generates the forgery probability map from the model."""
    model.eval()  # Ensure model is in eval mode
    
    # Prepare input tensor
    img_array = np.array(pil.resize((CFG.IMG_SIZE, CFG.IMG_SIZE)), np.float32) / 255.0
    x = torch.from_numpy(img_array).permute(2, 0, 1)[None].to(CFG.DEVICE)
    
    # Use autocast only if AMP is enabled and working properly
    if CFG.USE_AMP and CFG.SCALER is not None:
        with torch.amp.autocast(device_type=CFG.DEVICE.type, dtype=CFG.SCALER_DTYPE, enabled=True):
            pred = model.forward_seg(x, (CFG.IMG_SIZE, CFG.IMG_SIZE))
            prob = torch.sigmoid(pred)[0, 0].detach().cpu().numpy()
    else:
        pred = model.forward_seg(x, (CFG.IMG_SIZE, CFG.IMG_SIZE))
        prob = torch.sigmoid(pred)[0, 0].detach().cpu().numpy()
    
    # Clean up GPU memory
    del x, pred
    clear_memory(verbose=False)
    
    return prob

def enhanced_adaptive_mask(prob, alpha_grad=0.35):
    """Refines the probability map using gradient and adaptive thresholding."""
    
    # Ensure prob is in valid range [0, 1]
    prob = np.clip(prob, 0.0, 1.0)
    
    # Calculate gradient magnitude
    gx = cv2.Sobel(prob, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(prob, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_norm = grad_mag / (grad_mag.max() + 1e-6)
    
    # Enhance map with gradient information
    enhanced = (1 - alpha_grad) * prob + alpha_grad * grad_norm
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Adaptive Threshold: more conservative approach
    mean_val = np.mean(enhanced)
    std_val = np.std(enhanced)
    thr = max(mean_val + 0.3 * std_val, 0.15)  # Minimum threshold of 0.15
    
    mask = (enhanced > thr).astype(np.uint8)
    
    # Morphological Operations - slightly more aggressive cleaning
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    
    return mask, thr

def finalize_mask(prob, orig_size):
    """Resizes the segmented mask to the original image dimensions."""
    mask, thr = enhanced_adaptive_mask(prob)
    mask = cv2.resize(mask, orig_size, interpolation=cv2.INTER_NEAREST)
    return mask, thr

def pipeline_final(pil, model):
    """Full inference pipeline: segmentation, refinement, and final classification."""
    try:
        prob = segment_prob_map(pil, model)
        mask, thr = finalize_mask(prob, pil.size)
        
        area = int(mask.sum())
        
        # Calculate mean probability inside the detected mask area
        prob_resized = cv2.resize(prob, (mask.shape[1], mask.shape[0]))
        mean_inside = float(prob_resized[mask == 1].mean()) if area > 0 else 0.0
        
        # More balanced filtering conditions
        min_area = 200  # Reduced from 400 for smaller forgeries
        min_confidence = 0.25  # Reduced from 0.35 for more sensitivity
        
        # Final Classification/Filtering Condition
        if area < min_area or mean_inside < min_confidence:
            return "authentic", None, {"area": area, "mean_inside": mean_inside, "thr": thr}
            
        return "forged", mask, {"area": area, "mean_inside": mean_inside, "thr": thr}
        
    except Exception as e:
        print(f"Error in pipeline_final: {e}")
        return "authentic", None, {"area": 0, "mean_inside": 0.0, "thr": 0.0}