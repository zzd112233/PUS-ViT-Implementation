"""
Oberpfaffenhofen correlation figure (topic–scattering Pearson matrix).
Checkpoints: Ober/model, K=15, 7 classes.
Output: Ober/fig1_topic_scattering_correlation.pdf
"""

import os
import copy
import csv
import json

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["pdf.fonttype"] = 42

from VisionTransformer_7_RSD import VisionTransformer as ViT_RSD
from VisionTransformer_7_T import VisionTransformer as ViT_T
from PhysicsGuidedConvLayer import PhysicsGuidedConvLayer
from Train_Fusion import GuidedVisionTransformerT
import torch.nn as nn


class LegacyTopicModel(nn.Module):
    def __init__(self, embed_dim=64, num_topics=15, num_classes=7):
        super().__init__()
        self.topic_basis = nn.Parameter(torch.randn(num_topics, embed_dim))
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, num_topics))
        self.feature_to_topic = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_topics),
            nn.ReLU(),
        )

    def forward(self, cls_features):
        doc_topic = self.feature_to_topic(cls_features)
        doc_topic = F.relu(doc_topic)
        doc_topic = doc_topic / (doc_topic.sum(dim=1, keepdim=True) + 1e-10)
        recon = torch.matmul(doc_topic, self.topic_basis)
        return doc_topic, recon


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(ROOT, "Ober")
MODEL_DIR = os.path.join(SAVE_DIR, "model")
T_VAL = os.path.join(SAVE_DIR, "T_validation.txt")
RSD_VAL = os.path.join(SAVE_DIR, "RSD_validation.txt")
NUM_CLASSES = 7
NUM_TOPICS = 15
IMG_SIZE = 11
BATCH = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Lawn", "Forest", "Building", "Farmland",
    "Paved road", "Mixed area", "Bare land",
]
NAMES = [
    r"$\mathrm{Span}$", r"$P_S$", r"$P_D$", r"$P_V$",
    r"$P_2$", r"$P_3$", r"$\theta$", r"$\phi$",
    r"$x$", r"$y$", r"$a$", r"$b$",
]
PLAIN = [
    "Span", "PS", "PD", "PV", "P2", "P3",
    "theta", "phi", "x", "y", "a", "b",
]
MEANINGS = [
    "Total polarimetric power",
    "Power of surface scattering",
    "Power of double-bounce",
    "Power of volume scattering",
    "Power of 2nd component",
    "Power of 3rd component",
    "Twice the orientation angle",
    "Twice the helix angle",
    "Spherical prop. (2nd comp.)",
    "Spherical prop. (3rd comp.)",
    "Phase of T12 (2nd comp.)",
    "Phase of T12 (3rd comp.)",
]


def find_best(prefix):
    files = [f for f in os.listdir(MODEL_DIR) if f.startswith(prefix) and f.endswith(".pth")]
    scored = []
    for f in files:
        try:
            acc = float(f.rsplit("_", 1)[-1].replace(".pth", ""))
            scored.append((acc, f))
        except Exception:
            continue
    scored.sort(reverse=True)
    # Prefer high but not exact 100.00 when alternatives exist
    for acc, f in scored:
        if abs(acc - 100.0) > 1e-6:
            return os.path.join(MODEL_DIR, f)
    return os.path.join(MODEL_DIR, scored[0][1])


def load_models():
    """Ober SHAP models: single-branch legacy topic MLP, img_size=11, 7 classes."""
    pgcl_path = find_best("pgcl_model_best_")
    fusion_path = find_best("fusion_model_best_")
    print("Device:", DEVICE)
    print("PGCL:", pgcl_path)
    print("Fusion:", fusion_path)
    try:
        fusion_sd = torch.load(fusion_path, map_location=DEVICE, weights_only=False)
        pgcl_sd = torch.load(pgcl_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        fusion_sd = torch.load(fusion_path, map_location=DEVICE)
        pgcl_sd = torch.load(pgcl_path, map_location=DEVICE)

    K = int(fusion_sd["guidance_topic.topic_basis"].shape[0])
    C = int(fusion_sd["guidance_topic.class_prototypes"].shape[0])
    assert K == NUM_TOPICS and C == NUM_CLASSES, (K, C)

    guidance_topic = LegacyTopicModel(64, K, C).to(DEVICE)
    guidance_pgcl = PhysicsGuidedConvLayer(
        embed_dim=64, num_topics=K, num_classes=C, hidden_dim=128
    ).to(DEVICE)

    # T backbone without attaching dual-branch RSD layers
    base = ViT_T(
        img_size=IMG_SIZE,
        in_channels=10,
        embed_dim=64,
        depth=4,
        num_classes=NUM_CLASSES,
        num_topics=K,
        use_pgcl=False,
        rsd_model_path=None,
    ).to(DEVICE)

    model = GuidedVisionTransformerT(
        base, guidance_topic, guidance_pgcl, num_classes=NUM_CLASSES
    ).to(DEVICE)
    missing, unexpected = model.load_state_dict(fusion_sd, strict=False)
    print(f"Fusion load missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing:", missing[:8])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # quick topic shape check
    with torch.no_grad():
        t0, _ = model.guidance_topic(torch.zeros(1, 64, device=DEVICE))
        print("topic dim", tuple(t0.shape))
    return model, pgcl_path, fusion_path


def load_x(path):
    try:
        x = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        x = torch.load(path, map_location="cpu")
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return x


def load_paired(t_txt, rsd_txt):
    t_items, r_items = [], []
    with open(t_txt) as f:
        for line in f:
            p, lab = line.strip().rsplit(" ", 1)
            t_items.append((p, int(lab)))
    with open(rsd_txt) as f:
        for line in f:
            p, lab = line.strip().rsplit(" ", 1)
            r_items.append((p, int(lab)))
    assert len(t_items) == len(r_items)
    return t_items, r_items


def save_pdf(fig, name):
    path = os.path.join(SAVE_DIR, f"{name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("pdf:", name)


def write_csv(name, header, rows):
    with open(os.path.join(SAVE_DIR, name), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def topic_from_cls(topic_net, cls, mode="relu"):
    mods = list(topic_net.feature_to_topic.children())
    h = cls
    for m in mods[:-1]:
        h = m(h)
    if mode == "softmax":
        return F.softmax(h, dim=1)
    t = F.relu(h)
    return t / (t.sum(1, keepdim=True) + 1e-10)


@torch.no_grad()
def extract_cache(model, t_items, r_items):
    cls_all, topic_r, topic_s, gate_all = [], [], [], []
    scatter_all, labels, preds = [], [], []
    for i in range(0, len(t_items), BATCH):
        tb = t_items[i:i + BATCH]
        rb = r_items[i:i + BATCH]
        xt = torch.cat([load_x(p) for p, _ in tb], 0).to(DEVICE)
        labs = [lab for _, lab in tb]
        scs = [load_x(p).mean(dim=(2, 3)).squeeze(0).numpy() for p, _ in rb]
        logits, cls = model.backbone(xt)
        tr = topic_from_cls(model.guidance_topic, cls, "relu")
        ts = topic_from_cls(model.guidance_topic, cls, "softmax")
        gate = model.guidance_pgcl.physics_guidance(tr)
        fusion_logits, _ = model(xt)
        cls_all.append(cls.cpu()); topic_r.append(tr.cpu()); topic_s.append(ts.cpu())
        gate_all.append(gate.cpu())
        scatter_all.append(np.stack(scs, 0))
        labels.append(np.array(labs))
        preds.append(fusion_logits.argmax(1).cpu().numpy())
        if (i // BATCH) % 10 == 0:
            print(f"  extract {min(i+BATCH, len(t_items))}/{len(t_items)}")
    return {
        "cls": torch.cat(cls_all),
        "topic": torch.cat(topic_r),
        "topic_soft": torch.cat(topic_s),
        "gate": torch.cat(gate_all),
        "scatter": np.concatenate(scatter_all),
        "labels": np.concatenate(labels),
        "preds": np.concatenate(preds),
    }


def fig_mapping(cache):
    topics = cache["topic_soft"].numpy()
    scatter = cache["scatter"]
    K, C = topics.shape[1], scatter.shape[1]
    corr = np.zeros((K, C))
    for k in range(K):
        for c in range(C):
            tk, sc = topics[:, k], scatter[:, c]
            if tk.std() < 1e-12 or sc.std() < 1e-12:
                corr[k, c] = 0.0
            else:
                corr[k, c] = np.corrcoef(tk, sc)[0, 1]
    rows = []
    for k in range(K):
        cb = int(np.argmax(np.abs(corr[k])))
        rows.append([f"T{k}"] + [f"{corr[k,c]:.4f}" for c in range(C)]
                    + [f"{abs(corr[k,cb]):.4f}", PLAIN[cb], MEANINGS[cb]])
    write_csv("topic_quantity_corr.csv",
              ["Topic"] + PLAIN + ["max_abs_r", "dominant_quantity", "physical_meaning"], rows)
    write_csv("dominant_quantity.csv",
              ["Topic", "Dominant_Quantity", "PhysicalMeaning", "Correlation"],
              [[f"T{k}", PLAIN[int(np.argmax(np.abs(corr[k])))],
                MEANINGS[int(np.argmax(np.abs(corr[k])))],
                f"{corr[k, int(np.argmax(np.abs(corr[k])))]:.4f}"] for k in range(K)])

    plot_correlation_matrix(corr)
    return corr


# Manuscript Fig. 11 uses the same page box, axes, and colorbar as Fig. 16.
_FIG11_PAGE = (500.8443603515625 / 72.0, 385.279052734375 / 72.0)
_FIG11_AX = (
    17.440000534057617 / 500.8443603515625,
    (385.279052734375 - 369.83905029296875) / 385.279052734375,
    (426.64727783203125 - 17.440000534057617) / 500.8443603515625,
    (369.83905029296875 - 4.47906494140625) / 385.279052734375,
)
_FIG11_CAX = (
    439.9332275390625 / 500.8443603515625,
    (385.279052734375 - 369.83905029296875) / 385.279052734375,
    (458.20123291015625 - 439.9332275390625) / 500.8443603515625,
    (369.83905029296875 - 4.47906494140625) / 385.279052734375,
)
_FIG11_XLABELS = [
    r"$Span$", r"$P_{\mathrm{S}}$", r"$P_{\mathrm{D}}$", r"$P_{\mathrm{V}}$",
    r"$P_{2}$", r"$P_{3}$", r"$\theta$", r"$\varphi$",
    r"$x$", r"$y$", r"$a$", r"$b$",
]


def plot_correlation_matrix(corr, save_name="fig1_topic_scattering_correlation"):
    """Draw the Oberpfaffenhofen heatmap in the same layout as manuscript Fig. 11."""
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["DejaVu Serif"]
    matplotlib.rcParams["font.size"] = 10
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["axes.linewidth"] = 0.8
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["savefig.dpi"] = 100

    k_count, c_count = corr.shape
    fig = plt.figure(figsize=_FIG11_PAGE)
    ax = fig.add_axes(list(_FIG11_AX))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(c_count))
    ax.set_yticks(range(k_count))
    ax.set_xticklabels(_FIG11_XLABELS[:c_count], fontsize=10)
    ax.set_yticklabels([rf"$T_{{{k}}}$" for k in range(k_count)], fontsize=9)
    ax.tick_params(length=0, pad=4.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    for k in range(k_count):
        for c in range(c_count):
            value = corr[k, c]
            if abs(value) < 0.12:
                continue
            ax.text(
                c, k, f"{value:.2f}", ha="center", va="center", fontsize=6.5,
                color=("white" if abs(value) > 0.55 else "black"),
            )
    cax = fig.add_axes(list(_FIG11_CAX))
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    cbar.ax.tick_params(labelsize=8, width=0.8, length=3.5)
    cbar.outline.set_linewidth(0.8)
    cbar.set_label(r"Pearson $r$", fontsize=10)
    path = os.path.join(SAVE_DIR, f"{save_name}.pdf")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print("pdf:", path)


def fig_purity(cache):
    topics = cache["topic_soft"].numpy()
    labels = cache["labels"]
    K = topics.shape[1]
    dominant = topics.argmax(1)
    hard = np.zeros(K); soft = np.zeros(K); support = np.zeros(K, dtype=int)
    class_hist = np.zeros((K, NUM_CLASSES))
    for k in range(K):
        m = dominant == k
        support[k] = int(m.sum())
        if support[k] > 0:
            counts = np.bincount(labels[m], minlength=NUM_CLASSES).astype(float)
            class_hist[k] = counts / counts.sum()
            hard[k] = counts.max() / counts.sum()
        w = topics[:, k]
        if w.sum() > 1e-12:
            mass = np.array([w[labels == c].sum() for c in range(NUM_CLASSES)])
            soft[k] = (mass / (mass.sum() + 1e-12)).max()
    write_csv("topic_purity.csv",
              ["Topic", "Support", "HardPurity", "SoftPurity"],
              [[f"T{k}", int(support[k]), f"{hard[k]:.4f}", f"{soft[k]:.4f}"] for k in range(K)])
    order = np.argsort(-soft)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    x = np.arange(K)
    axes[0].bar(x - 0.2, hard[order], 0.4, label="Hard purity", color="#4C72B0", edgecolor="k", lw=0.3)
    axes[0].bar(x + 0.2, soft[order], 0.4, label="Soft purity", color="#55A868", edgecolor="k", lw=0.3)
    axes[0].axhline(1 / NUM_CLASSES, color="gray", ls="--", lw=1, label="chance")
    axes[0].set_xticks(x); axes[0].set_xticklabels([rf"$T_{{{k}}}$" for k in order], fontsize=8)
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("Purity")
    axes[0].set_title("Oberpfaffenhofen: Topic Purity"); axes[0].legend(fontsize=8)
    im = axes[1].imshow(class_hist[order].T, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    axes[1].set_xticks(range(K)); axes[1].set_xticklabels([rf"$T_{{{k}}}$" for k in order], fontsize=8)
    axes[1].set_yticks(range(NUM_CLASSES)); axes[1].set_yticklabels(CLASS_NAMES, fontsize=8)
    axes[1].set_title("Topic–Class Assignment")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_pdf(fig, "fig2_topic_purity")
    return hard, soft


def fig_gating(model, cache):
    cls = cache["cls"].to(DEVICE)
    topic = cache["topic"].to(DEVICE)
    gates = cache["gate"].numpy()
    labels = cache["labels"]
    N, H = gates.shape
    attr_list = []
    for i in range(0, N, BATCH):
        c = cls[i:i + BATCH].detach()
        t = topic[i:i + BATCH].detach()
        fused = torch.cat([c, t], 1)
        fused_out = model.guidance_pgcl.fusion_layer(fused).detach()
        gate = model.guidance_pgcl.physics_guidance(t).detach().requires_grad_(True)
        guided = model.guidance_pgcl.feature_enhancement(fused_out * gate)
        logits = model.fusion_head(torch.cat([c, guided], 1))
        pred = logits.argmax(1)
        logits[torch.arange(logits.size(0), device=DEVICE), pred].sum().backward()
        attr_list.append((gate.grad * gate).detach().abs().cpu().numpy())
    mean_attr = np.concatenate(attr_list, 0).mean(0)
    mean_gate = gates.mean(0)
    importance = np.argsort(-mean_attr)
    write_csv("gate_importance.csv",
              ["GateDim", "MeanGate", "GradxAct"],
              [[h, f"{mean_gate[h]:.6f}", f"{mean_attr[h]:.6e}"] for h in range(H)])

    def predict(gate_keep=None):
        with torch.no_grad():
            fused = torch.cat([cls, topic], 1)
            fused_out = model.guidance_pgcl.fusion_layer(fused)
            g = model.guidance_pgcl.physics_guidance(topic)
            if gate_keep is not None:
                g = g * torch.tensor(gate_keep, dtype=torch.float32, device=DEVICE).view(1, -1)
            guided = model.guidance_pgcl.feature_enhancement(fused_out * g)
            return model.fusion_head(torch.cat([cls, guided], 1))

    logits0 = predict()
    pred0 = logits0.argmax(1)
    fracs = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    abl = []
    for fr in fracs:
        keep = np.ones(H, np.float32)
        n = int(round(fr * H))
        keep[importance[:n]] = 0
        l1 = predict(keep)
        flip = (pred0 != l1.argmax(1)).float().mean().item() * 100
        abl.append((fr, flip))
        print(f"  zero top {int(fr*100)}% gates: flip {flip:.2f}%")
    rng = np.random.RandomState(0)
    rand = []
    for fr in fracs:
        vals = []
        for _ in range(2):
            keep = np.ones(H, np.float32)
            n = int(round(fr * H))
            if n > 0:
                keep[rng.choice(H, n, replace=False)] = 0
            l1 = predict(keep)
            vals.append((pred0 != l1.argmax(1)).float().mean().item() * 100)
        rand.append((fr, float(np.mean(vals)), float(np.std(vals))))

    topics_np = cache["topic"].numpy()
    active = np.where(topics_np.std(0) > 1e-8)[0]
    order = active[np.argsort(-topics_np[:, active].mean(0))] if len(active) else np.arange(min(5, topics_np.shape[1]))
    gsel = importance[:32]
    mat = np.zeros((len(order), len(gsel)))
    for i, k in enumerate(order):
        for j, h in enumerate(gsel):
            if topics_np[:, k].std() < 1e-12 or gates[:, h].std() < 1e-12:
                mat[i, j] = 0
            else:
                mat[i, j] = np.corrcoef(topics_np[:, k], gates[:, h])[0, 1]

    top_n = min(40, H)
    top_idx = importance[:top_n]
    fig = plt.figure(figsize=(12.5, 9.0))
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar(range(top_n), mean_attr[top_idx], color="#4C72B0", edgecolor="k", lw=0.2)
    ax0.set_ylabel(r"$|$Grad $\times$ Gate$|$")
    ax0.set_title(f"Oberpfaffenhofen: Gating Attribution (Top-{top_n})")
    ax0.set_xticks(range(top_n))
    ax0.set_xticklabels([str(i) for i in top_idx], fontsize=5.5, rotation=90)

    ax1 = fig.add_subplot(gs[0, 1])
    im = ax1.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax1.set_yticks(range(len(order)))
    ax1.set_yticklabels([rf"$T_{{{k}}}$" for k in order])
    ax1.set_xticks(range(len(gsel)))
    ax1.set_xticklabels([str(i) for i in gsel], fontsize=6, rotation=90)
    ax1.set_title(rf"Topic–Gate Correlation (active, $n={len(order)}$)")
    # annotate like fig1 topic–scattering heatmap
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if abs(v) >= 0.12:
                ax1.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.0,
                         color=("white" if abs(v) > 0.55 else "black"))
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot([x[0] for x in abl], [x[1] for x in abl], "o-", color="#C44E52", lw=2, label="Attr-guided")
    ax2.errorbar([x[0] for x in rand], [x[1] for x in rand], yerr=[x[2] for x in rand],
                 fmt="s--", color="#4C72B0", lw=2, capsize=3, label="Random")
    ax2.set_xlabel("Fraction of gate dims zeroed")
    ax2.set_ylabel("Flip rate (%)")
    ax2.set_title("Gating Ablation (Flip)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    class_gate = np.array([gates[labels == c].mean() if (labels == c).any() else 0
                           for c in range(NUM_CLASSES)])
    ax3.bar(range(NUM_CLASSES), class_gate, color="#8172B3", edgecolor="k", lw=0.3)
    ax3.set_xticks(range(NUM_CLASSES))
    ax3.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=8)
    ax3.set_ylabel("Mean gate activation")
    ax3.set_title("Class-wise Mean Gating Strength")
    fig.tight_layout()
    save_pdf(fig, "fig5_gating_attribution")
    return order, abl


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    model, pgcl_path, fusion_path = load_models()
    t_items, r_items = load_paired(T_VAL, RSD_VAL)
    print("samples:", len(t_items))
    cache = extract_cache(model, t_items, r_items)
    acc = 100.0 * (cache["preds"] == cache["labels"]).mean()
    print(f"Fusion Acc: {acc:.2f}%")
    print("native active:", int((cache["topic"].numpy().std(0) > 1e-8).sum()), "/15")
    print("softmax active:", int((cache["topic_soft"].numpy().std(0) > 1e-8).sum()), "/15")

    corr = fig_mapping(cache)
    print("Correlation matrix written to", SAVE_DIR)
    print("Max |r| =", float(np.max(np.abs(corr))))
    print("Done:", SAVE_DIR)
    print("Top correlations:", tops[:8])


if __name__ == "__main__":
    main()
