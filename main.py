import importlib
from utils import tool
from pathlib import Path
import os
import mlflow
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT) 

def main():
    cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "config.yaml")

    env = tool.detect_environment()
    cfg.globals.env = env
    print(f"🌍 Detected environment: {env.upper()}")
    print(f"📁 Project root: {PROJECT_ROOT}\n")

    stages = cfg.stages
    log_cfg = cfg.globals.log

    print(f"🧠 Launching pipeline with {len(stages)} stages\n")

    prev_ckpt_path = None   # last checkpoint path

    for stage in stages:
        name = (stage["name"]).lower()

        stage_cfg_path = PROJECT_ROOT / "configs" / "stages" / f"{name}.yaml"

        if not os.path.exists(stage_cfg_path):
            print(f"❌ Stage config not found: {stage_cfg_path}")
            continue

        stage_cfg = OmegaConf.load(stage_cfg_path)
        merged_cfg = OmegaConf.merge(cfg, stage_cfg)

        if prev_ckpt_path:
            merged_cfg.train.resume_from_checkpoint = prev_ckpt_path
            print(f"🔗 Using previous checkpoint: {prev_ckpt_path}")

        print(f"🚀 Stage: {name.upper()}")
        if 'description' in stage_cfg:
            print(f"📘 Description: {stage_cfg.description}")

        # Create output dir
        save_dir = Path(merged_cfg.train.get("save_dir", f"/runs/{name}_default"))
        save_dir = PROJECT_ROOT / save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Load trainer module, eg: train/train_dinov2.py
        trainer_module_name = f"train.train_{name}"
        try:
            trainer_module = importlib.import_module(trainer_module_name)
        except ModuleNotFoundError:
            print("❌ Trainer module not found: {trainer_module_name}")
            continue
        
        ckpt_path = None    # save best_model_path from the running stage
        # Run with MLflow logging
        if log_cfg.use_mlflow:
            tracking_uri = str(log_cfg.tracking_uri).replace("\\", "/")
            if not tracking_uri.startswith("file:"):
                tracking_uri = f"file:{PROJECT_ROOT / tracking_uri}"

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(log_cfg.experiment_name)
            print(f"🧾 MLflow tracking URI: {tracking_uri}")
            print(f"🧪 Experiment: {log_cfg.experiment_name}\n")

            if log_cfg.get("skip_finished_stages", True):
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
                    "epochs": merged_cfg.train.get("epochs", "N/A"),
                    "batch_size": merged_cfg.train.get("batch_size", "N/A"),
                    "learning_rate": merged_cfg.train.get("learning_rate", "N/A")
                })

                # train_dinov2.run(dinov_cfg)
                ckpt_path = trainer_module.run(merged_cfg)

                mlflow.log_artifact(str(save_dir))
                print(f"✅ Stage [{name}] completed and logged to MLflow.\n")
        else:
            ckpt_path = trainer_module.run(merged_cfg)
            print(f"✅ Stage [{name}] completed (no MLflow logging).\n")

        if isinstance(ckpt_path, str) and os.path.exists(ckpt_path):
            prev_ckpt_path = ckpt_path
            print(f"💾 Stored checkpoint for next stage: {prev_ckpt_path}")
        else:
            print(f"ℹ️ Stage [{name}] did not return a checkpoint (likely inference/demo). Keeping previous.")

    print("\n🎯 All stages completed successfully!\n")

if __name__ == "__main__":
    main()