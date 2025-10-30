import os
from pathlib import Path

from data_modules.datamodule_linear_probe import ImageNet100LinearProbeDataModule
from models.linear_probe_module import LinearProbeModule

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

def run(cfg):
    print("🚀 Starting Linear Probe training...")
    torch.autograd.set_detect_anomaly(True)
    print(cfg.train)

    train_cfg = cfg.train
    log_cfg = cfg.globals.log
    data_cfg = cfg.data

    save_dir = Path(train_cfg.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    seed_everything(train_cfg.seed)

    # ---- MLflow logger ----
    experiment_name = log_cfg.experiment_name
    tracking_uri = log_cfg.tracking_uri
    mlf_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=tracking_uri)
    tb_logger = TensorBoardLogger(save_dir="runs", name="linear_probe_exp")

    # ---- Checkpoint callback ----
    ckpt_cb = ModelCheckpoint(
        dirpath=save_dir,
        filename="linearprobe-{epoch:02d}-{val_acc1:.3f}",
        save_top_k=2,
        save_last=True,
        monitor="val/acc1",
        mode="max",
        auto_insert_metric_name=False
    )

    # ---- Learning rate monitor ----
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    # ---- EarlyStopping ----
    early_stop_cb = EarlyStopping(
        monitor="val/acc1",
        min_delta=train_cfg.early_stop_min_delta,
        patience=train_cfg.early_stop_patience,
        mode="max",
        verbose=True
    )

    # ---- Data + Model ----
    dm = ImageNet100LinearProbeDataModule(data_cfg, train_cfg)
    dm.setup()
    model = LinearProbeModule(cfg)

    # ---- Trainer ----
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=train_cfg.epochs,
        precision=train_cfg.precision,
        accumulate_grad_batches=train_cfg.accumulate_grad_batches,
        logger = [tb_logger, mlf_logger],
        callbacks=[ckpt_cb, lr_cb, early_stop_cb],
        log_every_n_steps=train_cfg.log_every_n_steps,
        default_root_dir=save_dir,
        gradient_clip_val=train_cfg.gradient_clip_val,
        deterministic=True,
        benchmark=False,
        fast_dev_run=False,
    )

    # ---- Resume or start fresh ----
    last_ckpt_path = os.path.join(save_dir, "last.ckpt")
    if os.path.exists(last_ckpt_path):
        print(f"🔄 Resuming training from: {last_ckpt_path}")
        ckpt_path = "last"
    else:
        print("🚀 No checkpoint found. Starting from scratch.")
        ckpt_path = None

    # ---- Fit ----
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)

    print(f"✅ Linear Probe stage completed. Best: {ckpt_cb.best_model_path}")

    return ckpt_cb.best_model_path