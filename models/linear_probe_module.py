from models.dino_module import DINOv2LightningModule
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearProbeModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        ckpt_path = cfg.model.ckpt_path

        # 1. load DINO encoder
        dinov2 = DINOv2LightningModule.load_from_checkpoint(ckpt_path, cfg=cfg)
        self.backbone = dinov2.backbone_s

        # 2. free backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        # 3. add linear head
        feat_dim = getattr(self.backbone, "embed_dim", 384)
        n_classes = cfg.model.num_classes
        self.classifier = nn.Linear(feat_dim, n_classes)

        # 4. loss func
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone.forward(x)
            if isinstance(feats, dict):
                feats=feats["x"] if "x" in feats else list(feats.values())

    def training_step(self, *args, **kwargs):
        return super().training_step(*args, **kwargs)
    
    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)
    
    def configure_optimizers(self):
        return super().configure_optimizers()