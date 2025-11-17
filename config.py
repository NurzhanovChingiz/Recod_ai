from utils.clear_gpu import clear_memory
from utils.set_seed import set_seed
from utils.summary import summary
import os
from torch import nn
import torch
from torch.backends import cudnn
from utils.loss.BCEFocalTverskyLoss import BCEFocalTverskyLoss
from utils.loss.ComputeSegLoss import ComputeSegLoss
from transformers import AutoImageProcessor, AutoModel
from utils.decoder import DinoTinyDecoder
from utils.segmenter import DinoSegmenter
from utils.unet_dino import DinoUNetDecoder

from torchvision.transforms import v2

class CFG: 
    SEED: int = 137
    set_seed(SEED)
    clear_memory()
    cudnn.benchmark = True

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAIN_DIR = os.getcwd()
    BASE_DIR = f"{MAIN_DIR}/recodai-luc-scientific-image-forgery-detection" # Must be pre-downloaded
    AUTH_DIR = f"{BASE_DIR}/train_images/authentic"
    FORG_DIR = f"{BASE_DIR}/train_images/forged"
    MASK_DIR = f"{BASE_DIR}/train_masks"
    TEST_DIR = f"{BASE_DIR}/test_images"
    DINO_PATH = f"{MAIN_DIR}/models/dinov3-89m-transformers-default-v1"# Must be pre-extracted from tar
    SAVE_MODEL: bool = False
    PATH_SAVE_MODEL = f"{MAIN_DIR}/output/model_seg_final.pt"
    TEST_SIZE = 0.2
    IMG_SIZE = 512
    BATCH_SIZE = 64 # Small batch size due to high resolution/ViT features
    EPOCHS = 1
    LR = 1e-5
    MOMENTUM = 0.937
    WEIGHT_DECAY = 0.0005
    # WEIGHT_DECAY *= 64 / 32   # Scale weight decay with batch size
    USE_AMP = True
    SCALER_DTYPE = torch.float16 if USE_AMP else None  # float16 or bfloat16
    SCALER = torch.amp.GradScaler(device=DEVICE, enabled=USE_AMP) if USE_AMP else None
    
    PROCESSOR = AutoImageProcessor.from_pretrained(DINO_PATH, local_files_only=True)
    ENCODER = AutoModel.from_pretrained(DINO_PATH, local_files_only=True).eval().to(DEVICE)
    
    HEAD = DinoTinyDecoder(in_ch=768, out_ch=1, bottleneck_ch=64).to(DEVICE)
    # HEAD = DinoUNetDecoder(in_ch=768, out_ch=1).to(DEVICE)
    
    MODEL = DinoSegmenter(HEAD, ENCODER, PROCESSOR).to(DEVICE)
    # summary(MODEL)

#     LOSS_FN = BCEFocalTverskyLoss(
#     bce_weight=0.5,   # 0.3–0.7, can tune
#     alpha=0.7,        # penalize FN more (good for tiny forgeries)
#     beta=0.3,
#     gamma=0.75        # focal strength; 0.75–1.5 common
# )
    # LOSS_FN = ComputeSegLoss(
    #     pixel_loss_type="blur_focal",
    #     use_dice=False,
    #     use_quality_dice=False,
    #     use_sigmoid_focal=False,
    #     use_quality_focal=False,
    #     w_pix=1.0,
    #     w_area=0.5,
    #     w_dist=0.2,
    #     gamma=1.5,
    #     alpha=0.25,
    #     blur_alpha=0.05,
    # )
    LOSS_FN = nn.BCEWithLogitsLoss()
    OPTIMIZER = torch.optim.AdamW(MODEL.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,betas=(MOMENTUM, 0.999))
    
    TRAIN_TRANSFORM = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    )
    ])
    TEST_TRANSFORM = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    )
    ])

# test for config
if __name__ == "__main__":
    print("Configuration:")
    # dirs check
    print("BASE_DIR exists:", os.path.exists(CFG.BASE_DIR))
    print("AUTH_DIR exists:", os.path.exists(CFG.AUTH_DIR))
    print("FORG_DIR exists:", os.path.exists(CFG.FORG_DIR))
    print("MASK_DIR exists:", os.path.exists(CFG.MASK_DIR))
    print("TEST_DIR exists:", os.path.exists(CFG.TEST_DIR))
    print("DINO_PATH exists:", os.path.exists(CFG.DINO_PATH))
    print('Output model path:', os.path.exists(CFG.PATH_SAVE_MODEL))