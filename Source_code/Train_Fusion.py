"""
Two-stage PUS-ViT training (Stage-1 RSD -> Stage-2 T).

Aligned with the revised manuscript (Table II / Section III):
  Stage-1: train BoW + topic + ViT + PGCL on 12-channel RSD(+Span) patches.
  Stage-2: freeze Stage-1 guidance copy (topic+PGCL readout); train T backbone,
           inserted PGCL+FC(128->64) adapters, fusion head FC(192->128->M), and KD.
"""

from __future__ import annotations

import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from VisionTransformer_7_RSD import VisionTransformer as ViT_RSD
from VisionTransformer_7_T import VisionTransformer as ViT_T


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path, data_root=None, base_dir=None):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    if data_root:
        return os.path.normpath(os.path.join(data_root, path))
    if base_dir:
        return os.path.normpath(os.path.join(base_dir, path))
    return os.path.normpath(path)


def load_patch_tensor(path: str, device) -> torch.Tensor:
    """Load a .pt patch as (1, C, H, W)."""
    tensor = torch.load(path, map_location="cpu")
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 4:
        raise RuntimeError(f"Unexpected tensor shape {tuple(tensor.shape)} in {path}")
    return tensor.to(device)


class TxtDataset(Dataset):
    """List file lines: `<tensor_path> <class_id>`."""

    def __init__(self, txt_path, data_root=None):
        self.data_root = data_root
        self.txt_dir = os.path.dirname(os.path.abspath(txt_path))
        samples = []
        with open(txt_path, "r", encoding="utf-8") as fh:
            for line in fh:
                items = line.strip().split()
                if len(items) >= 2:
                    samples.append((items[0], int(items[1])))
        if not samples:
            raise RuntimeError(f"No samples found in {txt_path}")
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel, label = self.samples[idx]
        if os.path.isabs(rel):
            path = rel
        elif self.data_root:
            path = os.path.join(self.data_root, rel)
            if not os.path.exists(path):
                path = os.path.join(self.txt_dir, rel)
        else:
            path = os.path.join(self.txt_dir, rel)
        return path, label


def build_dataloader(txt_path, batch_size, shuffle, data_root=None, num_workers=0):
    dataset = TxtDataset(txt_path, data_root=data_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def calc_ce(logits, labels, device):
    return nn.CrossEntropyLoss()(logits.to(device), labels.to(device))


def stack_batch(paths, device):
    chunks = [load_patch_tensor(p, device) for p in paths]
    return torch.cat(chunks, dim=0)


class GuidedVisionTransformerT(nn.Module):
    """
    Stage-2 wrapper:
      - backbone: T ViT with inserted PGCL adapters (trainable)
      - frozen guidance_topic / guidance_pgcl from Stage-1 (no grad)
      - fusion head: Concat(CLS_64, h_128)=192 -> FC(192 -> 128 -> M)
      - auxiliary backbone head supplies KD reference logits
    """

    def __init__(self, base_model, guidance_topic, guidance_pgcl, num_classes, fusion_dim=128):
        super().__init__()
        self.backbone = base_model
        self.guidance_topic = guidance_topic
        self.guidance_pgcl = guidance_pgcl
        self.num_classes = num_classes
        self.fusion_head = nn.Sequential(
            nn.Linear(self.backbone.embed_dim + fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, x):
        logits_ref, cls_features = self.backbone(x)
        with torch.no_grad():
            topic, _ = self.guidance_topic(cls_features)
            guided = self.guidance_pgcl(cls_features, topic)
        fusion_logits = self.fusion_head(torch.cat([cls_features, guided], dim=1))
        return fusion_logits, logits_ref


def train_rsd_with_pgcl(config):
    """Stage-1: train BoW + topic + ViT + PGCL (all trainable)."""
    device = config["device"]
    probe = build_dataloader(
        config["train_txt"], batch_size=1, shuffle=False, data_root=config.get("data_root")
    )
    actual_in_channels = None
    for paths, _ in probe:
        actual_in_channels = load_patch_tensor(paths[0], "cpu").shape[1]
        break
    if actual_in_channels is None:
        raise RuntimeError("Could not infer input channels from training patches.")
    print(f"Detected Stage-1 channels: {actual_in_channels} (expected 12)")

    model = ViT_RSD(
        embed_dim=64,
        num_classes=config["num_classes"],
        num_topics=config["num_topics"],
        use_pgcl=True,
        depth=config.get("rsd_depth", 3),
        mlp_dim=256,
        patch_size=7,
        in_channels=actual_in_channels,
        use_bow=config.get("use_bow", True),
        num_words=config.get("num_words", 64),
    ).to(device)
    print(
        "Stage-1 single-path: BoW -> topic; ViT CLS -> PGCL; head FC(128->M). "
        f"V={config.get('num_words', 64)}, K={config['num_topics']}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    train_loader = build_dataloader(
        config["train_txt"], config["batch_size"], True, config.get("data_root")
    )
    val_loader = build_dataloader(
        config["test_txt"], config["batch_size"], False, config.get("data_root")
    )

    best_acc, best_path = -1.0, None
    os.makedirs(config["save_dir"], exist_ok=True)
    print("=" * 60)
    print("Stage-1: train RSD + BoW + topic + PGCL")
    print("=" * 60)

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = total_cls = total_topic = 0.0
        steps = 0
        for paths, labels in train_loader:
            labels = labels.to(device)
            batch = stack_batch(paths, device)
            optimizer.zero_grad()
            logits, cls_features, _ = model(batch)
            cls_loss = calc_ce(logits, labels, device)
            if getattr(model, "use_bow", False):
                bow_feat, _ = model.bow_encoder(batch)
                topic_loss = model.topic_model.compute_supervised_loss(bow_feat, labels)
            else:
                topic_loss = model.topic_model.compute_supervised_loss(cls_features, labels)
            loss = cls_loss + config["topic_lambda"] * topic_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_cls += cls_loss.item()
            total_topic += topic_loss.item()
            steps += 1
        scheduler.step()
        print(
            f"[RSD][Epoch {epoch + 1}] "
            f"Loss={total_loss / max(steps, 1):.4f} "
            f"(Cls={total_cls / max(steps, 1):.4f}, Topic={total_topic / max(steps, 1):.4f})"
        )
        acc = evaluate(model, val_loader, device)
        if acc >= best_acc:
            best_acc = acc
            best_path = os.path.join(config["save_dir"], f"pgcl_model_best_{best_acc:.2f}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  Saved Stage-1 checkpoint: {best_path}")

    if best_path is None:
        raise RuntimeError("Stage-1 did not save a checkpoint.")
    model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"Stage-1 done. Best Acc={best_acc:.2f}%")

    # Frozen guidance copy for Stage-2
    guidance_topic = copy.deepcopy(model.topic_model).to(device).eval()
    guidance_pgcl = copy.deepcopy(model.pgcl).to(device).eval()
    for p in list(guidance_topic.parameters()) + list(guidance_pgcl.parameters()):
        p.requires_grad = False
    return guidance_topic, guidance_pgcl, best_path


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    preds, gts = [], []
    for paths, labels in data_loader:
        batch = stack_batch(paths, device)
        out = model(batch)
        logits = out[0] if isinstance(out, tuple) else out
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        gts.append(labels.numpy())
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    acc = round(float((preds == gts).mean() * 100.0), 2)
    print(f"  Eval Acc: {acc:.2f}%")
    return acc


def train_t_with_guidance(config, guidance_topic, guidance_pgcl, rsd_model_path):
    """Stage-2: freeze guidance copy; train backbone/adapters/fusion + KD."""
    device = config["device"]
    base_model = ViT_T(
        embed_dim=64,
        num_classes=config["num_classes"],
        num_topics=config["num_topics"],
        use_pgcl=False,
        depth=4,
        mlp_dim=256,
        patch_size=7,
        in_channels=config.get("t_in_channels", 10),
        rsd_model_path=rsd_model_path,
    ).to(device)

    guided = GuidedVisionTransformerT(
        base_model,
        guidance_topic.to(device),
        guidance_pgcl.to(device),
        num_classes=config["num_classes"],
    ).to(device)
    for p in list(guided.guidance_topic.parameters()) + list(guided.guidance_pgcl.parameters()):
        p.requires_grad = False
    print("Frozen Stage-1 guidance copy (topic + PGCL readout).")

    trainable = list(guided.backbone.parameters()) + list(guided.fusion_head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    train_loader = build_dataloader(
        config["train_txt"], config["batch_size"], True, config.get("data_root")
    )
    val_loader = build_dataloader(
        config["test_txt"], config["batch_size"], False, config.get("data_root")
    )

    best_acc, best_path = -1.0, None
    os.makedirs(config["save_dir"], exist_ok=True)
    print("=" * 60)
    print("Stage-2: train T backbone + inserted PGCL + fusion head + KD")
    print("=" * 60)

    for epoch in range(config["epochs"]):
        guided.train()
        # Keep guidance modules in eval mode
        guided.guidance_topic.eval()
        guided.guidance_pgcl.eval()
        total_loss, steps = 0.0, 0
        for paths, labels in train_loader:
            labels = labels.to(device)
            batch = stack_batch(paths, device)
            optimizer.zero_grad()
            fusion_logits, logits_ref = guided(batch)
            ce = calc_ce(fusion_logits, labels, device)
            kd = nn.functional.kl_div(
                torch.log_softmax(fusion_logits, dim=1),
                torch.softmax(logits_ref.detach(), dim=1),
                reduction="batchmean",
            )
            loss = ce + config["kd_lambda"] * kd
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        scheduler.step()
        print(f"[T][Epoch {epoch + 1}] Loss={total_loss / max(steps, 1):.4f}")
        acc = evaluate_fusion(guided, val_loader, device)
        if acc >= best_acc:
            best_acc = acc
            best_path = os.path.join(config["save_dir"], f"fusion_model_best_{best_acc:.2f}.pth")
            torch.save(guided.state_dict(), best_path)
            print(f"  Saved Stage-2 checkpoint: {best_path}")

    if best_path is None:
        raise RuntimeError("Stage-2 did not save a checkpoint.")
    print(f"Stage-2 done. Best Acc={best_acc:.2f}%")
    return best_path


@torch.no_grad()
def evaluate_fusion(model, data_loader, device):
    model.eval()
    preds, gts = [], []
    for paths, labels in data_loader:
        batch = stack_batch(paths, device)
        fusion_logits, _ = model(batch)
        preds.append(torch.argmax(fusion_logits, dim=1).cpu().numpy())
        gts.append(labels.numpy())
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    acc = round(float((preds == gts).mean() * 100.0), 2)
    print(f"  Fusion Eval Acc: {acc:.2f}%")
    return acc


def load_yaml_config(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Please `pip install PyYAML` to use --config") from exc
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_argparser():
    p = argparse.ArgumentParser(description="Train PUS-ViT (two-stage RSD -> T)")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--rsd_train", type=str, default=None)
    p.add_argument("--rsd_val", type=str, default=None)
    p.add_argument("--t_train", type=str, default=None)
    p.add_argument("--t_val", type=str, default=None)
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--stage", type=str, default="both", choices=["both", "rsd", "t"])
    p.add_argument("--rsd_ckpt", type=str, default=None)
    return p


def merge_config(args):
    cfg = load_yaml_config(args.config)
    for key in ["data_root", "save_dir", "num_classes", "epochs", "batch_size", "device", "seed"]:
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    if args.rsd_train:
        cfg["rsd_train"] = args.rsd_train
    if args.rsd_val:
        cfg["rsd_val"] = args.rsd_val
    if args.t_train:
        cfg["t_train"] = args.t_train
    if args.t_val:
        cfg["t_val"] = args.t_val

    cfg.setdefault("seed", 42)
    cfg.setdefault("batch_size", 32)
    cfg.setdefault("epochs", 50)
    cfg.setdefault("lr", 4.0e-4)
    cfg.setdefault("num_classes", 9)
    cfg.setdefault("num_topics", 15)
    cfg.setdefault("rsd_depth", 3)
    cfg.setdefault("topic_lambda", 0.1)
    cfg.setdefault("kd_lambda", 0.2)
    cfg.setdefault("rsd_in_channels", 12)
    cfg.setdefault("t_in_channels", 10)
    cfg.setdefault("use_bow", True)
    cfg.setdefault("num_words", 64)
    cfg.setdefault("save_dir", "checkpoints")
    cfg.setdefault("data_root", None)

    device_str = cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    cfg["device"] = torch.device(device_str)

    data_root = cfg.get("data_root")
    rsd_train = resolve_path(cfg.get("rsd_train"), data_root)
    rsd_val = resolve_path(cfg.get("rsd_val"), data_root) or rsd_train
    t_train = resolve_path(cfg.get("t_train"), data_root)
    t_val = resolve_path(cfg.get("t_val"), data_root) or t_train
    cfg["save_dir"] = resolve_path(cfg.get("save_dir"), base_dir=os.getcwd())

    shared = {
        "device": cfg["device"],
        "batch_size": cfg["batch_size"],
        "epochs": cfg["epochs"],
        "lr": cfg["lr"],
        "num_classes": cfg["num_classes"],
        "num_topics": cfg["num_topics"],
        "save_dir": cfg["save_dir"],
        "data_root": data_root,
    }
    cfg["rsd"] = {
        **shared,
        "rsd_depth": cfg["rsd_depth"],
        "topic_lambda": cfg["topic_lambda"],
        "rsd_in_channels": cfg["rsd_in_channels"],
        "use_bow": cfg["use_bow"],
        "num_words": cfg["num_words"],
        "train_txt": rsd_train,
        "test_txt": rsd_val,
    }
    cfg["t"] = {
        **shared,
        "kd_lambda": cfg["kd_lambda"],
        "t_in_channels": cfg["t_in_channels"],
        "train_txt": t_train,
        "test_txt": t_val,
    }
    return cfg


def load_guidance_from_ckpt(ckpt_path, cfg):
    probe = ViT_RSD(
        embed_dim=64,
        num_classes=cfg["num_classes"],
        num_topics=cfg["num_topics"],
        use_pgcl=True,
        depth=cfg.get("rsd_depth", 3),
        in_channels=cfg.get("rsd_in_channels", 12),
        use_bow=True,
        num_words=cfg.get("num_words", 64),
    )
    probe.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    topic = copy.deepcopy(probe.topic_model).eval()
    pgcl = copy.deepcopy(probe.pgcl).eval()
    for p in list(topic.parameters()) + list(pgcl.parameters()):
        p.requires_grad = False
    return topic, pgcl


if __name__ == "__main__":
    args = build_argparser().parse_args()
    cfg = merge_config(args)
    set_seed(int(cfg.get("seed", 42)))

    print("PUS-ViT training")
    print(f"  device={cfg['device']}, save_dir={cfg['save_dir']}")
    print(f"  Stage-1 lists: {cfg['rsd']['train_txt']} | val={cfg['rsd']['test_txt']}")
    print(f"  Stage-2 lists: {cfg['t']['train_txt']} | val={cfg['t']['test_txt']}")
    print(f"  BoW={cfg['rsd']['use_bow']}, K={cfg['num_topics']}")

    if args.stage in ("both", "rsd"):
        guidance_topic, guidance_pgcl, rsd_best_path = train_rsd_with_pgcl(cfg["rsd"])
    else:
        if not args.rsd_ckpt:
            raise SystemExit("--rsd_ckpt is required when --stage t")
        rsd_best_path = args.rsd_ckpt
        guidance_topic, guidance_pgcl = load_guidance_from_ckpt(rsd_best_path, cfg)

    if args.stage in ("both", "t"):
        train_t_with_guidance(cfg["t"], guidance_topic, guidance_pgcl, rsd_best_path)
