from typing import List
import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16
import timm
from timm.models.vision_transformer import resize_pos_embed


class DINOv2LightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg

        # backbone (student/teacher)
        self.backbone_s, feat_dim = build_backbone(cfg.model.backbone)
        self.backbone_t, feat_dim = build_backbone(cfg.model.backbone)
        for p in self.backbone_t.parameters():
            p.requires_grad=False
        
        # heads (student/teacher)
        proj_dim = cfg.model.projection_dim
        hidden_dim = cfg.model.hidden_dim
        self.head_s = MLPHead(feat_dim, hidden_dim=hidden_dim, out_dim=proj_dim)
        self.head_t = MLPHead(feat_dim, hidden_dim=hidden_dim, out_dim=proj_dim)
        for p in self.head_t.parameters():
            p.requires_grad=False

        # loss
        self.loss_fn = DINOLoss(
            out_dim=proj_dim,
            warmup_teacher_temp=cfg.model.temperature_teacher_warmup,
            teacher_temp=cfg.model.temperature_teacher,
            warmup_steps=int(cfg.train.warmup_epochs * 1000),  # 粗略按步数近似
            total_steps=int(cfg.train.epochs* 1000),
            center_momentum=cfg.model.center_momentum,
        )

        self.base_lr = cfg.train.learning_rate
        self.weight_decay = cfg.train.weight_decay
        self.momentum_teacher_start = cfg.train.momentum_teacher

        # log
        self.example_input_array = torch.randn(2, 3, cfg.data.global_size, cfg.data.global_size)
    
    def forward_backbone(self, model, x):
        """
        兼容 timm ViT 的 forward_backbone，支持 multi-crop 输入 (224 / 96)
        """
        B, C, H, W = x.shape
        patch = model.patch_embed.patch_size
        ph, pw = (patch if isinstance(patch, tuple) else (patch, patch))
        gh, gw = H // ph, W // pw
        cur_tokens = gh * gw
        num_prefix = getattr(model, "num_prefix_tokens", 1)

        # 确保 strict_img_size=False
        if hasattr(model.patch_embed, "strict_img_size"):
            model.patch_embed.strict_img_size = False

        # 调整 pos_embed
        if hasattr(model, "pos_embed") and isinstance(model.pos_embed, nn.Parameter):
            pe = model.pos_embed
            base_tokens = pe.shape[1] - num_prefix
            if base_tokens != cur_tokens:
                with torch.no_grad():
                    # 生成一个假 token tensor 作为目标 shape
                    dummy = torch.zeros(1, num_prefix + cur_tokens, pe.shape[-1])
                    new_pe = resize_pos_embed(pe, dummy, num_prefix_tokens=num_prefix)
                    model.pos_embed = nn.Parameter(new_pe)

        # 走标准的 timm 特征提取
        feats = model.forward_features(x)

        # timm 有些模型返回 dict
        if isinstance(feats, dict):
            for k in ["x", "last_hidden_state", "features"]:
                if k in feats:
                    feats = feats[k]
                    break

        if not isinstance(feats, torch.Tensor):
            raise ValueError(f"Unexpected features type: {type(feats)}")

        # 返回 CLS（等价于你之前的 feats[:, 0]）
        if feats.dim() == 3:
            return feats[:, 0]
        elif feats.dim() == 2:
            return feats
        else:
            raise ValueError(f"Unexpected feature shape: {feats.shape}")
    
    @torch.no_grad()
    def _update_teacher(self, m):
        # EMA update only for same-shaped params
        for ps, pt in zip(self.backbone_s.parameters(), self.backbone_t.parameters()):
            if ps.data.shape == pt.data.shape:
                pt.data.mul_(m).add_(ps.data, alpha=(1.0 - m))
        for ps, pt in zip(self.head_s.parameters(), self.head_t.parameters()):
            if ps.data.shape == pt.data.shape:
                pt.data.mul_(m).add_(ps.data, alpha=(1.0 - m))

        for ms, mt in zip(self.head_s.modules(), self.head_t.modules()):
            if isinstance(ms, nn.BatchNorm1d):
                mt.running_mean.data.copy_(ms.running_mean.data)
                mt.running_var.data.copy_(ms.running_var.data)

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
            t_out = [t.detach() for t in t_out]

        try:
            loss = self.loss_fn(s_out, t_out)
        except Exception as e:
            print(f"❌ Loss computation failed at step {self.global_step}: {e}")
            self.trainer.should_stop = True
            return torch.tensor(0.0, requires_grad=True, device=self.device)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        self.log("train/loss_epoch_smooth", loss.detach(), prog_bar=False, on_epoch=True, sync_dist=True)

        # update teacher
        with torch.no_grad():
            m = self._teacher_momentum()
            self._update_teacher(m)
            self.log("train/momentum_teacher", m, on_step=True, prog_bar=False)
        
        # ---- NaN 防护机制 ----
        if torch.isnan(loss):
            print(f"🚨 NaN detected at step {self.global_step}, stopping training safely.")
            self.trainer.should_stop = True
            # 返回一个干净的标量，防止反向传播崩溃
            return torch.tensor(0.0, requires_grad=True, device=loss.device)
        
        if not torch.isfinite(loss):
            self.log("train/nan_step", self.global_step)
            print(f"🚨 Non-finite loss at step {self.global_step}, skipping batch.")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)
        # cosine schedule with warmup (per epoch stepping OK for simplicity)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.train.epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
    
    def on_after_backward(self):
        """自适应梯度裁剪，在每次反向传播后执行"""
        # 过滤掉无梯度参数
        grads = [p.grad.norm() for p in self.parameters() if p.grad is not None]
        if len(grads) == 0:
            return
        # 计算 90% 分位梯度范数
        clip_value = torch.quantile(torch.stack(grads), 0.9).item()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_value)
        self.log("train/grad_clip_value", clip_value, on_step=True, prog_bar=False)

        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.log("train/grad_norm", grad_norm, on_step=True, prog_bar=False)
    
    def forward(self, x):
        # 用 student backbone + head 做一次完整的前向传播
        z = self.forward_backbone(self.backbone_s, x)
        out = self.head_s(z)
        return out
    
    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        feats = self.forward_backbone(self.backbone_s, imgs)
        dummy_loss = torch.tensor(0.0, device=self.device)
        self.log("val/loss", dummy_loss)

# ---------------------------
# Projection Head
# --------------------------- 
class MLPHead(nn.Module):
    """
    Linear → GELU → LayerNorm → Linear → GELU → LayerNorm → Linear → LayerNorm(affine=False)
    """
    def __init__(self, in_dim, hidden_dim=4096, out_dim=256, num_layers=3, last_bn=True):
        super().__init__()
        layers = []

        # dim_list : input and output dim of each layer
        # eg: [768, 4096, 4096, 256]
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
                nn.Linear(dim_list[i], dim_list[i+1], bias=False),  # 全连接层，线性变换
                nn.GELU(),                                # 激活函数（比 ReLU 更平滑）
                nn.LayerNorm(dim_list[i+1], elementwise_affine=False)           # 批归一化，稳定训练
            ]

        # 最后一层线性层：hidden_dim → out_dim
        layers += [nn.Linear(dim_list[-2], dim_list[-1], bias=False)]

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
        teacher_logits = (teacher_logits - self.center).clamp(-50, 50)
        teacher_prob = F.softmax(teacher_logits / t, dim=-1).detach()

        # student logits (for all views)
        student_logits = torch.cat(student_out, dim=0).clamp(-50, 50)  # [B*N_views, dim]
        student_logprob = F.log_softmax(student_logits, dim=-1)

        # Cross-entropy between each student view and each teacher view
        # For simplicity: average over all pairings (N_views × N_global)
        n_global = len(teacher_out)
        n_views = len(student_out)
        B = teacher_out[0].shape[0]

        # expand teacher to match student repeats per (view)
        teacher_prob_rep = teacher_prob.repeat(n_views // n_global + (1 if n_views % n_global else 0), 1)[:B * n_views]
        loss = -torch.sum(teacher_prob_rep * student_logprob, dim=-1).mean()
        loss = loss + 1e-8

        # update center
        with torch.no_grad():
            self.update_center(torch.cat(teacher_out, dim=0))

        return loss

# ---------------------------
# Backbones
# ---------------------------
def build_backbone(name: str):
    """
    change to TIMM, Vision Transformer backbone
    make 96X96 and 224X224 both are acceptable 
    """
    m = timm.create_model(name, pretrained=False)
    m.set_grad_checkpointing(True)

    # accept dynamic image size
    if hasattr(m, "patch_embed") and hasattr(m.patch_embed, "strict_img_size"):
        m.patch_embed.strict_img_size = False

    feat_dim = getattr(m, "embed_dim",768)
    
    return m, feat_dim
    
