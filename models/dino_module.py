from typing import List
import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16


class DINOv2LightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg

        # backbone (student/teacher)
        self.backbone_s, feat_dim = build_backbone(cfg.model.get("backbone", "vit_base_patch16"))
        self.backbone_t, feat_dim = build_backbone(cfg.model.get("backbone", "vit_base_patch16"))
        for p in self.backbone_t.parameters():
            p.requires_grad=False
        
        # heads (student/teacher)
        proj_dim = cfg.model.get("projection_dim", 256)
        hidden_dim = cfg.model.get("hidden_dim", 4096)
        self.head_s = MLPHead(feat_dim, hidden_dim=hidden_dim, out_dim=proj_dim)
        self.head_t = MLPHead(feat_dim, hidden_dim=hidden_dim, out_dim=proj_dim)
        for p in self.head_t.parameters():
            p.requires_grad=False

        # loss
        self.loss_fn = DINOLoss(
            out_dim=proj_dim,
            warmup_teacher_temp=cfg.model.get("temperature_teacher", 0.04),
            teacher_temp=cfg.model.get("temperature_teacher", 0.04),
            warmup_steps=int(cfg.train.get("warmup_epochs", 10) * 1000),  # 粗略按步数近似
            total_steps=int(cfg.train.get("epochs", 100) * 1000),
            center_momentum=cfg.model.get("center_momentum", 0.9),
        )

        self.base_lr = cfg.train.get("learning_rate", 0.0003)
        self.weight_decay = cfg.train.get("weight_decay", 0.0001)
        self.momentum_teacher_start = cfg.train.get("momentum_teacher", 0.996)

        # log
        self.example_input_array = torch.randn(2, 3, cfg.data.get("image_size", 224), cfg.data.get("image_size", 224))
    
    def forward_backbone(self, model, x):
        feats = model._process_input(x)        # patch embedding
        B = feats.shape[0]
        cls_token = model.class_token.expand(B, -1, -1)
        feats = torch.cat([cls_token, feats], dim=1)
        feats = model.encoder(feats)           # Transformer encoder
        return feats[:, 0]                     # 取 CLS token 作为视觉特征
    
    @torch.no_grad()
    def _update_teacher(self, m):
        # EMA for teacher (cosine schedule from m0 -> 1.0)
        for ps, pt in zip(self.backbone_s.parameters(), self.backbone_t.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=(1.0 - m))
        for ps, pt in zip(self.head_s.parameters(), self.head_t.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=(1.0 - m))

    def _teacher_momentum(self):
        # cosine schedule (m_t) from m0 to 1
        cur = self.global_step
        tot = max(1, self.trainer.estimated_stepping_batches)
        return 1.0 - (1.0 - self.momentum_teacher_start) * (math.cos(math.pi * cur / tot) + 1) / 2

    def training_step(self, batch, batch_idx):
        # batch = (list_of_crops, _)
        views, _ = batch  # views: list[Tensor] len = n_global+n_local
        # student on all views
        s_out = []
        for v in views:
            z = self.forward_backbone(self.backbone_s, v)
            s_out.append(self.head_s(z))

        # teacher only on global views（前2个）
        with torch.no_grad():
            gviews = views[:2]
            t_out = []
            for v in gviews:
                zt = self.forward_backbone(self.backbone_t, v)
                t_out.append(self.head_t(zt))

        loss = self.loss_fn(s_out, t_out)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # update teacher
        with torch.no_grad():
            m = self._teacher_momentum()
            self._update_teacher(m)
            self.log("train/momentum_teacher", m, on_step=True, prog_bar=False)

        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)
        # cosine schedule with warmup (per epoch stepping OK for simplicity)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.train.get("epochs", 100))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

    def forward(self, x):
        # 用 student backbone + head 做一次完整的前向传播
        z = self.forward_backbone(self.backbone_s, x)
        out = self.head_s(z)
        return out
    
# ---------------------------
# Projection Head
# --------------------------- 
class MLPHead(nn.Module):
    """
    MLPHead (投影头)
    -----------------
    作用：
        将 backbone（如 ViT / ResNet）输出的通用视觉特征，
        通过多层非线性映射 (MLP) 投射到对比学习的“嵌入空间”中。
        对比损失（NT-Xent、DINO loss）在这个投影空间里计算。

    示例：
        MLPHead(in_dim=768, hidden_dim=4096, out_dim=256, num_layers=3)
        # 输入 768 → 输出 256
        # 结构: Linear(768→4096) → GELU → BN → Linear(4096→4096) → GELU → BN → Linear(4096→256)
    """
    def __init__(self, in_dim, hidden_dim=4096, out_dim=256, num_layers=3, last_bn=True):
        super().__init__()
        layers = []

        # dim_list 定义每层的输入输出维度列表
        # 举例: [768, 4096, 4096, 256]
        # 含义：输入维度 in_dim=768，经过两层 hidden_dim=4096，最后输出到 out_dim=256

        # dim_list = [2048] + [4096] * (3 - 1) + [256]
        # dim_list = [2048] + [4096, 4096] + [256]
        # dim_list = [2048, 4096, 4096, 256]
        dim_list = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]   

        # 构建前 num_layers-1 层，每层包含：Linear → GELU → BatchNorm
        # 举例: Linear(768→4096) → GELU → BN(4096)
        for i in range(len(dim_list) - 2):
            # Linear → GELU → BatchNorm
            layers += [
                nn.Linear(dim_list[i], dim_list[i+1]),  # 全连接层，线性变换
                nn.GELU(),                                # 激活函数（比 ReLU 更平滑）
                nn.BatchNorm1d(dim_list[i+1])           # 批归一化，稳定训练
            ]

        # 最后一层线性层：hidden_dim → out_dim
        layers += [nn.Linear(dim_list[-2], dim_list[-1])]

        # DINO 原论文建议最后加一个不带 affine 参数的 BatchNorm
        # affine=False 表示不学习 gamma/beta，只做标准化
        if last_bn:
            layers +=[nn.BatchNorm1d(dim_list[-1], affine=False)]
        
        # 将所有层顺序组合成一个完整的神经网络
        # 例如: x → Linear1 → GELU → BN → Linear2 → GELU → BN → Linear3 → BN → z
        self.net = nn.Sequential(*layers)

    """
    前向传播：
        输入：backbone 提取的图像特征 (B, in_dim)
        输出：投影到嵌入空间后的向量 (B, out_dim)
    """
    def forward(self, x):
        return self.net(x)

class DINOLoss(nn.Module):
    def __init__(self, out_dim, warmup_teacher_temp=0.04, teacher_temp=0.04,
                 warmup_steps=0, total_steps=100000, center_momentum=0.9):
        super().__init__()
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp = warmup_teacher_temp
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.center_momentum = center_momentum
        self.step = 0

    def _teacher_temp_now(self):
        if self.step < self.warmup_steps:
            # linear warmup
            progress = self.step / self.warmup_steps
            return self.warmup_teacher_temp + progress * (self.teacher_temp - self.warmup_teacher_temp)
        return self.teacher_temp
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        # teacher_output: (B*#global_views, dim)
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(self, student_out: List[torch.Tensor], teacher_out: List[torch.Tensor]):
        """
        student_out: list of tensors from all crops   (N_views x [B, dim])
        teacher_out: list of tensors from global only (N_global x [B, dim])
        """
        self.step += 1
        t = self._teacher_temp_now()

        # teacher probs (sharpen + center)
        teacher_logits = torch.cat(teacher_out, dim=0) # [B*N_global, dim]
        teacher_logits = (teacher_logits - self.center)
        teacher_prob = F.softmax(teacher_logits, dim=-1).detach()

        # student logits (for all views)
        student_logits = torch.cat(student_out, dim=0)  # [B*N_views, dim]
        student_logprob = F.log_softmax(student_logits, dim=-1)

        # Cross-entropy between each student view and each teacher view
        # For simplicity: average over all pairings (N_views × N_global)
        n_global = len(teacher_out)
        n_views = len(student_out)
        B = teacher_out[0].shape[0]

        # expand teacher to match student repeats per (view)
        teacher_prob_rep = teacher_prob.repeat(n_views // n_global + (1 if n_views % n_global else 0), 1)[:B * n_views]
        loss = -torch.sum(teacher_prob_rep * student_logprob, dim=-1).mean()

        # update center
        with torch.no_grad():
            self.update_center(torch.cat(teacher_out, dim=0))

        return loss

# ---------------------------
# Backbones
# ---------------------------
def build_backbone(name: str):
    name = (name or "vit_base_patch16").lower()
    if "vit" in name:
        m = vit_b_16(weights=None)
        feat_dim = 768 # torchvision vit_b_16 classifier out
        return m, feat_dim
    else:
        # fallback
        m = vit_b_16(weights=None)
        return m, 768
    
