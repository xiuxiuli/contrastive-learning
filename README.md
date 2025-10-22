🧩 Contrastive Learning MLOps Pipeline - IN PROGRESS

End-to-End Multi-Stage Self-Supervised Learning Pipeline built with Hydra, MLflow, and PyTorch Lightning.
Designed for modular experimentation, reproducibility, and scalable model development.

🚀 Overview

This project implements a multi-stage visual contrastive learning pipeline, integrating state-of-the-art self-supervised learning (SSL) techniques such as DINOv2, Linear Probe evaluation, and multi-modal CLIP-style alignment.
The architecture follows modern MLOps best practices, combining Hydra configuration management, MLflow experiment tracking, and PyTorch Lightning for modular and reproducible deep learning workflows.

🧠 Pipeline Stages
Stage	Description	Output
Stage 1 — DINOv2 Pretraining	Self-supervised representation learning on ImageNet-100.	Vision backbone with strong features.
Stage 2 — Linear Probe	Train a linear classifier on frozen DINOv2 embeddings to measure representation quality.	Linear evaluation accuracy.
Stage 3 — CLIP Training	Multi-modal contrastive alignment between image and text features.	Unified vision-language embedding space.
Stage 4 — Demo / Retrieval	Launch retrieval demo for visual-semantic search.	End-to-end inference pipeline.

Each stage runs independently and can be resumed, skipped, or reconfigured dynamically through the unified Hydra + MLflow system.

⚙️ Technical Stack
Category	Technologies
Core Framework	PyTorch • PyTorch Lightning • torchvision
Experiment Management	Hydra • MLflow • YAML-based config groups
Architecture	DINOv2 • Linear Probe • CLIP
Data Handling	Hugging Face Datasets (ImageNet-100)
Pipeline Control	Hydra Compose + Dynamic Overrides
MLOps	MLflow Tracking Server • Artifact logging • Stage-level skip via MLflow run state
Utilities	tqdm • omegaconf • pyyaml • pathlib
🧩 Features

🔄 Multi-Stage Pipeline — DINOv2 → Linear Probe → CLIP → Demo

⚙️ Hydra Configuration System — dynamic config composition & inheritance

📊 MLflow Integration — full tracking of parameters, metrics, and artifacts

🧠 Idempotent Stage Execution — skip completed stages using MLflow run status

🧩 Lightning Training Framework — clean modular structure and reproducibility

☁️ Scalable & Extensible — ready for multi-GPU, Colab, or cloud training

🧭 Project Structure
contrastive-learning/
│
├── main.py                    # Hydra + MLflow pipeline controller
├── configs/
│   ├── config.yaml             # Global pipeline configuration
│   └── stages/
│       ├── dinov2.yaml
│       ├── linear_probe.yaml
│       ├── clip.yaml
│       └── demo.yaml
│
├── train/
│   ├── train_dinov2.py
│   ├── train_linear_probe.py
│   ├── train_clip.py
│   └── train_demo.py
│
├── utils/
│   └── tool.py
└── requirements.txt

🧮 Example Run
# Run full pipeline
python main.py

# Or run only one stage
python main.py stages=clip

# Override parameters dynamically
python main.py stages=dinov2 train.epochs=50 train.batch_size=128


Hydra automatically saves merged configs under:

outputs/<timestamp>/.hydra/config.yaml


MLflow automatically logs:

Parameters (epochs, lr, batch_size)

Metrics (loss, accuracy)

Artifacts (checkpoints, logs)

Run status (FINISHED / FAILED)


📊 MLflow UI

To visualize all experiments:

mlflow ui

Then open: http://127.0.0.1:5000


🧠 Summary

This project demonstrates how a modern machine learning engineer can architect a scalable, modular, and fully reproducible deep learning pipeline,
unifying model development and MLOps practices — from configuration and training, to logging, evaluation, and deployment.