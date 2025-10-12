import torch
from datasets.augumentations import build_simclr_transform


from models.simclr import SimCLR
from models.losses import NTXentLoss

def run(cfg):
    print("[INFO] Starting SimCLR training...")

    # 1. env setting
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    epochs = cfg["train"]["epochs"]
    batch_size = cfg["train"]["batch_size"]
    lr = cfg["train"]["lr"]
    temperature = cfg["train"]["temperature"]
    num_workers = cfg["train"]["num_workers"]
    dataset_root = cfg["dataset"]["root"]
    encoder_type = cfg["model"]["base_encoder"]
    projection_dim = cfg["model"]["projection_dim"]

    # 2. data augumentation and dataloader
    transform = build_simclr_transform("./config/augmentation_config.yaml")
    dataset = datasets.ImageFolder(root=dataset_root, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    print(f"[INFO] Dataset loaded from: {dataset_root}")
    print(f"[INFO] Total samples: {len(dataset)}, Batch size: {batch_size}")

    # 3. model, loss , optimizer
    model = SimCLR(base_encoder=encoder_type, projection_dim=projection_dim).to(device)
    criterion = NTXentLoss(temperature=temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"[INFO] Model initialized with encoder={encoder_type}")

    # 4. train loop
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch}/{epochs}]")

        for (x_i, x_j), _ in progress_bar:
            x_i, x_j = x_i.to(device), x_j.to(device)

            # 前向传播
            z_i, z_j = model(x_i, x_j)
            loss = criterion(z_i, z_j)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch}] Average Loss: {avg_loss:.4f}")

    # save encoder .pth
    os.makedirs("./checkpoints", exist_ok=True)
    encoder_path = "./checkpoints/encoder.pth"
    torch.save(model.encoder.state_dict(), encoder_path)
    print(f"[INFO] Encoder weights saved to {encoder_path} ✅")

