import argparse
import json
import random
from pathlib import Path


def positive_integer(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Train the SIMS text-aligned motion encoder.")
    parser.add_argument("--motion_file", nargs="+", required=True)
    parser.add_argument("--allowed_skills", nargs="+", default=["loco", "sitdown", "liedown"])
    parser.add_argument("--log_dir", default="output/motionclip")
    parser.add_argument("--num_epochs", type=positive_integer, default=200)
    parser.add_argument("--batch_size", type=positive_integer, default=256)
    parser.add_argument("--save_interval", type=positive_integer, default=50)
    parser.add_argument("--latent_dim", type=positive_integer, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--extract_from_text", action="store_true")
    return parser.parse_args()


def train_epoch(model, iterator, optimizer, device):
    import torch

    model.train()
    totals = {}
    for batch in iterator:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad()
        prediction, latent, text_latent, text_reconstruction = model(
            batch["motion_vector"], batch["mask"], batch["clip_features"],
        )
        loss, losses = model.compute_loss(
            batch["motion_vector"], prediction, latent, batch["clip_features"],
            batch["mask"], text_latent, text_reconstruction, batch["extra_clip_features"],
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss; check motion inputs and CLIP embeddings.")
        loss.backward()
        optimizer.step()
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + value
    return {key: value / len(iterator) for key, value in totals.items()}


def main():
    args = parse_args()
    import isaacgym
    import numpy as np
    import torch
    from torch.utils.data import ConcatDataset, DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from sims.learning.sims_motion_enc_datasets import MotionClipDataset
    from sims.learning.sims_motion_enc_model import MotionCLIP
    from sims.utils.motion_lib_dict import MotionLibDict, concat_dict_list

    if args.lr <= 0 or args.latent_dim % 4:
        raise ValueError("lr must be positive and latent_dim must be divisible by four.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu for an encoder smoke test.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    datasets = []
    for filename in args.motion_file:
        for skill, motion_dict in MotionLibDict._fetch_motion_files(filename).items():
            if skill not in args.allowed_skills or not motion_dict["motion_files"]:
                continue
            motion_dict["motion_info"] = concat_dict_list(motion_dict["motion_info"])
            datasets.append(MotionClipDataset(
                motion_dict, device=device, repeat={"sitdown": 2, "liedown": 4}.get(skill, 1),
                extract_from_text=args.extract_from_text,
            ))
    if not datasets:
        raise ValueError("No motions match allowed_skills; check your manifests and skill names.")
    iterator = DataLoader(ConcatDataset(datasets), batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = MotionCLIP(
        latent_dim=args.latent_dim, loss_weight={"recon": 1, "clip_mse": 0.01, "cosine": 10},
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    with SummaryWriter(str(log_dir)) as writer, (log_dir / "training.log").open("w") as logfile:
        for epoch in range(1, args.num_epochs + 1):
            losses = train_epoch(model, iterator, optimizer, device)
            line = "Epoch {}: {}".format(epoch, losses)
            print(line, flush=True)
            print(line, file=logfile, flush=True)
            for key, value in losses.items():
                writer.add_scalar("Loss/" + key, value, epoch)
            if epoch % args.save_interval == 0 or epoch == args.num_epochs:
                torch.save(model.state_dict(), str(log_dir / "epoch_{:04d}.pth".format(epoch)))
                torch.save(model.state_dict(), str(log_dir / "latest.pth"))
            writer.flush()


if __name__ == "__main__":
    main()
