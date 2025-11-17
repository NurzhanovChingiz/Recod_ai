from torch import nn
import torch
import math

class DinoSegmenter(nn.Module):
    def __init__(self, head, encoder, processor):
        super().__init__()
        self.encoder, self.processor = encoder, processor
        for p in self.encoder.parameters(): 
            p.requires_grad = False
        self.seg_head = head

    def forward_features(self,x):
        imgs = x
        inputs = self.processor(images=list(imgs), return_tensors="pt").to(x.device)
        with torch.no_grad(): 
            feats = self.encoder(**inputs).last_hidden_state
        B,N,C = feats.shape
        # Skip class token (index 0) and register tokens (last 4 tokens)
        # Only keep the spatial patch tokens
        num_register_tokens = 4
        patch_feats = feats[:,1:N-num_register_tokens,:]  # Skip [CLS] and register tokens
        fmap = patch_feats.permute(0,2,1)
        
        # Calculate spatial dimensions from remaining tokens
        num_patches = patch_feats.shape[1]
        s = int(math.sqrt(num_patches))
        assert s * s == num_patches, f"Expected perfect square, got {num_patches} patches"
        
        fmap = fmap.reshape(B,C,s,s)
        return fmap

    def forward_seg(self, x, target_size=None):
        fmap = self.forward_features(x)
        if target_size is None:
            # Default to input size if not specified
            target_size = (x.shape[2], x.shape[3])
        return self.seg_head(fmap, target_size)