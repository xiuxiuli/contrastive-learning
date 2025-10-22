"""
main.py — Unified Hydra + Lightning + MLflow Pipeline
Author: Xiuxiu Li
------------------------------------------------------
Controls multi-stage workflow:
DINOv2 → Linear Probe → CLIP → Demo
"""
import hydra
import importlib
from omegaconf import DictConfig
from utils import tool
from pathlib import Path
import os
import mlflow

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    stages = cfg.stages
    log_cfg = cfg.globals.log

    print(f"\n🧠 Launching Hydra-driven pipeline with {len(stages)} stages\n")

    for stage in stages:
        name = (stage["name"]).lower()
        stage_cfg = hydra.compose(config_name=f"stages/{name}").stages

        print(f"🚀 Stage: {name.upper()}")
        if 'description' in stage_cfg:
            desc = stage_cfg.description
            print(f"📘 Description: {desc}")

        # Create output dir
        root_dir = tool.get_root_dir(cfg)
        save_dir = Path(stage_cfg.train.save_dir if "train" in stage_cfg else "runs/default")
        save_dir = os.path.join(root_dir, save_dir)
        os.makedirs(save_dir, exist_ok=True)

        # Load corresponding trainer module, eg: train/train_dinov2.py
        trainer_module_name = f"train.train_{name}"
        try:
            trainer_module = importlib.import_module(trainer_module_name)
        except ModuleNotFoundError:
            print("❌ Trainer module not found: {trainer_module_name}")
            continue

        # Run with MLflow logging
        if log_cfg.use_mlflow:
            mlflow.set_tracking_uri(log_cfg.tracking_uri)
            mlflow.set_experiment(log_cfg.experiment_name)

            finish_runs = mlflow.search_runs(
                filter_string=f"tags.stage='{name}' and attributes.status='FINISHED'",
                experiment_names=[log_cfg.experiment_name]
            )

            if not finish_runs.empty:
                print(f"✅ Stage [{name}] already finished successfully — skipping.\n")
                continue
                
            #  mlflow.start_run(run_name="dinov2")
            with mlflow.start_run(run_name=name):
                mlflow.log_params({
                    "stage": name,
                    "epochs": stage_cfg.train.get("epochs", "N/A"),
                    "batch_size": stage_cfg.train.get("batch_size", "N/A"),
                    "learning_rate": stage_cfg.train.get("learning_rate", "N/A")
                })

                # train_dinov2.run(dinov_cfg)
                trainer_module.run(stage_cfg)
                mlflow.log_artifact(str(save_dir))
                print(f"✅ Stage [{name}] completed and logged to MLflow.\n")
        else:
            trainer_module.run(stage_cfg)
            print(f"✅ Stage [{name}] completed (no MLflow logging).\n")

    print("\n🎯 All stages completed successfully!\n")

if __name__ == "__main__":
    main()