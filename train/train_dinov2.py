import os
from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data.datamodule_dino import ImageNet100DinoDataModule
from models.dino_module import DINOv2LightningModule

def run(cfg):
    print("🚀 Starting DINOv2 training...")
    print(cfg.train)

    seed_everything(cfg.train.seed)

    log_cfg = cfg.globals.log
    experiment_name = log_cfg.experiment_name
    tracking_uri = log_cfg.tracking_uri

    save_dir = Path(cfg.train.get("save_dir", "runs/dinov2_exp1"))
    os.makedirs(save_dir, exist_ok=True)

    # ---- MLflow logger
    mlf_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=tracking_uri)

    # Lightning callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=save_dir,
        filename="dinov2-{epoch:02d}-{train_loss_epoch:.3f}",
        save_top_k=2,
        monitor="train/loss_epoch",
        mode="min",
        auto_insert_metric_name=False
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    # Data + Model
    dm = ImageNet100DinoDataModule(cfg.data, cfg.train)
    dm.setup()
    model = DINOv2LightningModule(cfg)

    # trainer
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=cfg.train.get("epochs", 100),
        precision="16-mixed",
        logger=mlf_logger,
        callbacks=[ckpt_cb, lr_cb],
        log_every_n_steps=20,
        default_root_dir=save_dir,
    )

    # Auto Resume
    print("🔍 Checking for existing checkpoint ...")
    last_ckpt_path = os.path.join(save_dir, "last.ckpt")

    if os.path.exists(last_ckpt_path):
        print(f"🔄 Resuming training from: {last_ckpt_path}")
        ckpt_path = "last"           # Lightning 自动检测路径
    else:
        print("🚀 No checkpoint found. Starting from scratch.")
        ckpt_path = None

    # fit
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)

    print(f"✅ DINOv2 stage completed. Best: {ckpt_cb.best_model_path}")

    # 让下游阶段能读取最佳权重路径
    return ckpt_cb.best_model_path


