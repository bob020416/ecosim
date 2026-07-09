import argparse
import datetime
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, WandbLogger
except ModuleNotFoundError:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, WandbLogger

_VBD_ROOT = Path(__file__).resolve().parents[1]
if str(_VBD_ROOT) not in sys.path:
    sys.path.insert(0, str(_VBD_ROOT))

from vbd.data.dataset import WaymaxDataset
from vbd.model.behavior_vae import BehaviorVAE

"""
cd third_party/VBD
python script/train_behavior_vae.py -cfg config/BehaviorVAE_full.yaml \
  --epochs 200 --num_gpus 2

python script/train_behavior_vae.py -cfg config/BehaviorVAE_full.yaml \
  --train_data_path data/overfit_sample_cl --val_data_path data/overfit_sample_cl \
  --epochs 500 --num_gpus 1 --batch_size 1 --overfit_batches 10 --num_workers 0 --log_every_n_steps 1 \
  --train_context_encoder 0 --latent_sampling 0 --kl_weight 0 --val_latent_sampling 0
"""


def load_config(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def _build_trainer(cfg: Dict[str, Any], output_path: str):
    num_gpus = int(cfg.get("num_gpus", 1))
    use_gpu = torch.cuda.is_available() and num_gpus != 0

    accelerator = "gpu" if use_gpu else "cpu"
    devices = num_gpus if use_gpu else 1

    use_wandb = bool(cfg.get("use_wandb", False))
    if use_wandb:
        logger = WandbLogger(
            name=os.path.basename(output_path),
            project=cfg.get("project"),
            entity=cfg.get("username"),
            log_model=False,
            dir=output_path,
        )
    else:
        logger = CSVLogger(output_path, name="BehaviorVAE", version=1, flush_logs_every_n_steps=50)

    overfit_batches = int(cfg.get("overfit_batches", 0))
    if overfit_batches < 0:
        overfit_batches = 0

    precision = cfg.get("precision", 32)
    if isinstance(precision, str) and precision.isdigit():
        precision = int(precision)

    trainer = pl.Trainer(
        num_nodes=int(cfg.get("num_nodes", 1)),
        max_epochs=int(cfg.get("epochs", 1)),
        accelerator=accelerator,
        devices=devices,
        strategy="auto",
        enable_progress_bar=True,
        logger=logger,
        enable_model_summary=True,
        detect_anomaly=False,
        gradient_clip_val=float(cfg.get("gradient_clip_val", 1.0)),
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=0,
        precision=precision,
        log_every_n_steps=int(cfg.get("log_every_n_steps", 100)),
        overfit_batches=overfit_batches if overfit_batches > 0 else 0,
        callbacks=[
            ModelCheckpoint(
                dirpath=output_path,
                save_top_k=5,
                save_weights_only=False,
                monitor="val/loss",
                filename="epoch={epoch:03d}",
                auto_insert_metric_name=False,
                every_n_epochs=1,
                save_on_train_epoch_end=False,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
    )
    return trainer


def train(cfg: Dict[str, Any]):
    pl.seed_everything(int(cfg.get("seed", 42)))
    torch.set_float32_matmul_precision("high")

    train_dataset = WaymaxDataset(
        data_dir=cfg["train_data_path"],
        anchor_path=cfg["anchor_path"],
    )
    val_dataset = WaymaxDataset(
        data_dir=cfg.get("val_data_path", cfg["train_data_path"]),
        anchor_path=cfg["anchor_path"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.get("batch_size", 1)),
        pin_memory=torch.cuda.is_available(),
        num_workers=int(cfg.get("num_workers", 0)),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.get("batch_size", 1)),
        pin_memory=torch.cuda.is_available(),
        num_workers=int(cfg.get("num_workers", 0)),
        shuffle=False,
    )

    output_root = cfg.get("log_dir", "output_behavior_vae")
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = cfg.get("model_name", "BehaviorVAE")
    run_name = f"{model_name}_{timestamp}"
    output_path = os.path.join(output_root, run_name)
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "config.yaml"), "w") as file:
        yaml.safe_dump(cfg, file, sort_keys=False)

    model = BehaviorVAE(cfg=cfg)

    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path:
        print("Load checkpoint:", ckpt_path)
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["state_dict"] if "state_dict" in state else state, strict=False)

    trainer = _build_trainer(cfg, output_path)
    trainer.fit(model, train_loader, val_loader, ckpt_path=cfg.get("init_from"))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--cfg", type=str, default="config/BehaviorVAE_full.yaml")

    # Common overrides
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--val_data_path", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    # Backward-compatible alias (maps to z_dim)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--z_dim", type=int, default=None)
    parser.add_argument("--agents_len", type=int, default=None)
    parser.add_argument("--future_len", type=int, default=None)
    parser.add_argument("--action_len", type=int, default=None)
    parser.add_argument("--train_context_encoder", type=int, default=None)
    parser.add_argument("--overfit_batches", type=int, default=None)
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--log_every_n_steps", type=int, default=None)
    parser.add_argument("--gradient_clip_val", type=float, default=None)

    # Model structure / ablations
    parser.add_argument("--use_context", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--interaction_layers", type=int, default=None)
    parser.add_argument("--interaction_heads", type=int, default=None)
    parser.add_argument("--decoder_mlp_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--truncate_after_invalid", type=int, default=None)

    # VAE / latent controls (int flags for easy CLI use: 0/1)
    parser.add_argument("--use_vae", type=int, default=None)
    parser.add_argument("--latent_sampling", type=int, default=None)
    parser.add_argument("--latent_sampling_warmup_steps", type=int, default=None)
    parser.add_argument("--val_latent_sampling", type=int, default=None)
    parser.add_argument("--kl_weight", type=float, default=None)
    parser.add_argument("--kl_warmup_steps", type=int, default=None)
    parser.add_argument("--free_nats", type=float, default=None)
    parser.add_argument("--use_conditional_prior", type=int, default=None)
    parser.add_argument("--use_z_scene", type=int, default=None)
    parser.add_argument("--z_scene_dim", type=int, default=None)
    return parser


def load_cfg(args) -> Dict[str, Any]:
    cfg = load_config(args.cfg)
    for key, value in vars(args).items():
        if key == "cfg":
            continue
        if value is not None:
            if key == "precision" and isinstance(value, str) and value.isdigit():
                value = int(value)
            if key == "latent_dim":
                cfg["z_dim"] = value
                continue
            cfg[key] = value
    return cfg


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_cfg(args)
    train(cfg)
