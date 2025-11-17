import torch
from torch import nn
class BinaryFocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1.0):
        """
        alpha > beta → penalize FN more than FP, good for small forged regions.
        gamma > 1 → focal behavior (focus on hard examples).
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits, targets: (B, 1, H, W)
        probs = torch.sigmoid(logits)
        probs = probs.view(logits.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1.0 - targets)).sum(dim=1)
        fn = ((1.0 - probs) * targets).sum(dim=1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        focal_tversky = (1.0 - tversky) ** self.gamma
        return focal_tversky.mean()

class BCEFocalTverskyLoss(nn.Module):
    def __init__(self, bce_weight=0.5, alpha=0.7, beta=0.3, gamma=0.75, eps=1e-6):
        """
        Compound loss:
        loss = w * BCEWithLogits + (1-w) * FocalTversky

        bce_weight ~0.3–0.7 is usually safe.
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()
        self.focal_tversky = BinaryFocalTverskyLoss(
            alpha=alpha, beta=beta, gamma=gamma
        )

    def forward(self, logits, targets):
        loss_bce = self.bce(logits, targets)
        loss_ft = self.focal_tversky(logits, targets)
        return self.bce_weight * loss_bce + (1.0 - self.bce_weight) * loss_ft + self.eps  # tiny constant to avoid zero loss