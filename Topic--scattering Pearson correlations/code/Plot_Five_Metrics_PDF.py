"""
Draw the Barnaul diagnostics as PDF files:
1) topic–scattering correlation
2) topic purity
3) prior shuffling
4) intervention
5) gating attribution

Checkpoints: K=15 fusion and Stage-1 models in Barnaul/model
"""

import os
import copy
import csv
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["pdf.fonttype"] = 42

from VisionTransformer_7_RSD import VisionTransformer as ViT_RSD
from VisionTransformer_7_T import VisionTransformer as ViT_T
from SupervisedTopicModelForPGCL import SupervisedTopicModelForPGCL
from Train_Fusion import GuidedVisionTransformerT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(ROOT, "Barnaul")
MODEL_DIR = os.path.join(SAVE_DIR, "model")
T_VAL = os.path.join(SAVE_DIR, "T_validation.txt")
RSD_VAL = os.path.join(SAVE_DIR, "RSD_validation.txt")
NUM_CLASSES = 9
NUM_TOPICS = 15
BATCH = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Forest", "Water", "Town", "Farmland", "Vegetation2",
    "Building", "Vegetation1", "Mixed area", "Flats",
]
NAMES = [
    r"$\mathrm{Span}$", r"$P_S$", r"$P_D$", r"$P_V$",
    r"$P_2$", r"$P_3$", r"$\theta$", r"$\phi$",
    r"$x$", r"$y$", r"$a$", r"$b$",
]


def find_best(prefix):
    files = [f for f in os.listdir(MODEL_DIR) if f.startswith(prefix) and f.endswith(".pth")]
    files.sort(key=lambda x: float(x.rsplit("_", 1)[-1].replace(".pth", "")), reverse=True)
    return os.path.join(MODEL_DIR, files[0])


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
    path = os.path.join(SAVE_DIR, name)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def load_models():
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
    assert K == NUM_TOPICS, K

    rsd = ViT_RSD(embed_dim=64, num_classes=NUM_CLASSES, num_topics=K,
                  use_pgcl=True, use_dual_branch=True).to(DEVICE)
    rsd.load_state_dict(pgcl_sd, strict=False)
    rsd.eval()

    guidance_topic = copy.deepcopy(rsd.fusion_pgcl.branch2_topic_model).to(DEVICE).eval()
    guidance_pgcl = copy.deepcopy(rsd.fusion_pgcl.branch2_pgcl).to(DEVICE).eval()

    base = ViT_T(embed_dim=64, num_classes=NUM_CLASSES, num_topics=K,
                 use_pgcl=False, rsd_model_path=pgcl_path).to(DEVICE)
    model = GuidedVisionTransformerT(base, guidance_topic, guidance_pgcl, NUM_CLASSES).to(DEVICE)
    missing, unexpected = model.load_state_dict(fusion_sd, strict=False)
    print(f"Fusion load missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, pgcl_path, fusion_path


def topic_from_cls(topic_net, cls, mode="relu"):
    """mode='relu': native; mode='softmax': Softmax(last Linear) for full K activity."""
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
        scs = []
        for p, _ in rb:
            x = load_x(p)
            scs.append(x.mean(dim=(2, 3)).squeeze(0).numpy())

        logits, cls = model.backbone(xt)
        # guidance uses no_grad topic inside forward; extract explicitly
        tr = topic_from_cls(model.guidance_topic, cls, "relu")
        ts = topic_from_cls(model.guidance_topic, cls, "softmax")
        gate = model.guidance_pgcl.physics_guidance(tr)
        fusion_logits, _ = model(xt)

        cls_all.append(cls.cpu())
        topic_r.append(tr.cpu())
        topic_s.append(ts.cpu())
        gate_all.append(gate.cpu())
        scatter_all.append(np.stack(scs, 0))
        labels.append(np.array(labs))
        preds.append(fusion_logits.argmax(1).cpu().numpy())
        if (i // BATCH) % 15 == 0:
            print(f"  extract {min(i+BATCH, len(t_items))}/{len(t_items)}")

    return {
        "cls": torch.cat(cls_all),
        "topic": torch.cat(topic_r),          # native for intervention
        "topic_soft": torch.cat(topic_s),     # for mapping/purity figures
        "gate": torch.cat(gate_all),
        "scatter": np.concatenate(scatter_all),
        "labels": np.concatenate(labels),
        "preds": np.concatenate(preds),
    }


@torch.no_grad()
def predict_from_cache(model, cls, topic=None, gate_mode="normal", gate_keep=None):
    """Light forward from cached CLS (+ optional topic override)."""
    cls = cls.to(DEVICE)
    if topic is None:
        topic = topic_from_cls(model.guidance_topic, cls, "relu")
    else:
        topic = topic.to(DEVICE)

    fused = torch.cat([cls, topic], 1)
    fused_out = model.guidance_pgcl.fusion_layer(fused)
    gate = model.guidance_pgcl.physics_guidance(topic)
    if gate_mode == "ones":
        gate = torch.ones_like(gate)
    elif gate_mode == "zeros":
        gate = torch.zeros_like(gate)
    elif gate_mode == "shuffle":
        gate = gate[torch.randperm(gate.size(0), device=gate.device)]
    elif gate_mode == "random":
        gate = torch.rand_like(gate)
    if gate_keep is not None:
        gate = gate * gate_keep.to(DEVICE).view(1, -1)
    guided = model.guidance_pgcl.feature_enhancement(fused_out * gate)
    fused_feat = torch.cat([cls, guided], 1)
    logits = model.fusion_head(fused_feat)
    return logits, gate


def effect(logits0, logits1, labels):
    p0 = F.softmax(logits0, 1)
    p1 = F.softmax(logits1, 1)
    pred0 = logits0.argmax(1)
    pred1 = logits1.argmax(1)
    flip = (pred0 != pred1).float().mean().item() * 100
    conf0 = p0.gather(1, pred0.unsqueeze(1)).squeeze(1)
    conf1 = p1.gather(1, pred0.unsqueeze(1)).squeeze(1)
    conf_drop = (conf0 - conf1).mean().item()
    kl = (p0 * (torch.log(p0 + 1e-12) - torch.log(p1 + 1e-12))).sum(1).mean().item()
    acc = (pred1 == labels).float().mean().item() * 100
    return acc, flip, conf_drop, kl


# Figure 16 of the manuscript: page box and axes measured from Fig16.pdf.
_FIG16_PAGE = (500.8443603515625 / 72.0, 385.279052734375 / 72.0)
_FIG16_AX = (
    17.440000534057617 / 500.8443603515625,
    (385.279052734375 - 369.83905029296875) / 385.279052734375,
    (426.64727783203125 - 17.440000534057617) / 500.8443603515625,
    (369.83905029296875 - 4.47906494140625) / 385.279052734375,
)
_FIG16_CAX = (
    439.9332275390625 / 500.8443603515625,
    (385.279052734375 - 369.83905029296875) / 385.279052734375,
    (458.20123291015625 - 439.9332275390625) / 500.8443603515625,
    (369.83905029296875 - 4.47906494140625) / 385.279052734375,
)
_FIG16_XLABELS = [
    r"$Span$", r"$P_{\mathrm{S}}$", r"$P_{\mathrm{D}}$", r"$P_{\mathrm{V}}$",
    r"$P_{2}$", r"$P_{3}$", r"$\theta$", r"$\varphi$",
    r"$x$", r"$y$", r"$a$", r"$b$",
]


def plot_correlation_matrix(corr, save_name="fig1_topic_scattering_correlation"):
    """Draw the Barnaul heatmap in the same layout as manuscript Fig. 16."""
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["DejaVu Serif"]
    matplotlib.rcParams["font.size"] = 10
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["axes.linewidth"] = 0.8
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["savefig.dpi"] = 100

    k_count, c_count = corr.shape
    fig = plt.figure(figsize=_FIG16_PAGE)
    ax = fig.add_axes(list(_FIG16_AX))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(c_count))
    ax.set_yticks(range(k_count))
    ax.set_xticklabels(_FIG16_XLABELS[:c_count], fontsize=10)
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
    cax = fig.add_axes(list(_FIG16_CAX))
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    cbar.ax.tick_params(labelsize=8, width=0.8, length=3.5)
    cbar.outline.set_linewidth(0.8)
    cbar.set_label(r"Pearson $r$", fontsize=10)
    path = os.path.join(SAVE_DIR, f"{save_name}.pdf")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print("pdf:", save_name)


# -------------------- 1) Mapping --------------------
def fig_mapping(cache):
    print("\n[1] Topic–scattering correlation")
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

    write_csv("metric1_topic_scattering_corr.csv",
              ["Topic"] + [f"Q{c}" for c in range(C)],
              [[f"T{k}"] + [f"{corr[k,c]:.4f}" for c in range(C)] for k in range(K)])

    plot_correlation_matrix(corr)
    return corr


# -------------------- 2) Purity --------------------
def fig_purity(cache):
    print("\n[2] Topic purity")
    topics = cache["topic_soft"].numpy()
    labels = cache["labels"]
    K = topics.shape[1]
    dominant = topics.argmax(1)
    hard = np.zeros(K)
    soft = np.zeros(K)
    support = np.zeros(K, dtype=int)
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

    write_csv("metric2_topic_purity.csv",
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
    axes[0].set_title("Topic Purity"); axes[0].legend(fontsize=8)
    im = axes[1].imshow(class_hist[order].T, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    axes[1].set_xticks(range(K)); axes[1].set_xticklabels([rf"$T_{{{k}}}$" for k in order], fontsize=8)
    axes[1].set_yticks(range(NUM_CLASSES)); axes[1].set_yticklabels(CLASS_NAMES, fontsize=8)
    axes[1].set_title("Topic–Class Assignment")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_pdf(fig, "fig2_topic_purity")


# -------------------- 3) Shuffling --------------------
def fig_shuffling(model, cache, n_rep=3):
    print("\n[3] Prior shuffling")
    cls = cache["cls"]
    topic = cache["topic"]
    labels = torch.tensor(cache["labels"], device=DEVICE)
    logits0, _ = predict_from_cache(model, cls, topic)
    base_acc = (logits0.argmax(1) == labels).float().mean().item() * 100
    results = [("Baseline", base_acc, 0.0, 0.0, 0.0, 0.0)]
    print(f"  Baseline Acc {base_acc:.2f}")

    def eval_cond(name, make_topic=None, gate_mode="normal", reps=1):
        accs, fs, cs, ks = [], [], [], []
        for _ in range(reps):
            t = topic if make_topic is None else make_topic(topic.clone())
            l1, _ = predict_from_cache(model, cls, t, gate_mode=gate_mode)
            a, f, c, k = effect(logits0, l1, labels)
            accs.append(a); fs.append(f); cs.append(c); ks.append(k)
        row = (name, float(np.mean(accs)), float(np.std(accs)),
               float(np.mean(fs)), float(np.mean(cs)), float(np.mean(ks)))
        results.append(row)
        print(f"  {name}: Acc {row[1]:.2f} flip {row[3]:.2f}")

    eval_cond("Shuffle topics (sample)",
              lambda t: t[torch.randperm(t.size(0))], reps=n_rep)
    eval_cond("Permute topic dims",
              lambda t: t[:, torch.randperm(t.size(1))], reps=n_rep)
    eval_cond("Random Dirichlet topics",
              lambda t: torch.distributions.Dirichlet(torch.ones(t.size(1))).sample((t.size(0),)),
              reps=n_rep)
    eval_cond("Noise topics",
              lambda t: (lambda r: r / (r.sum(1, keepdim=True) + 1e-10))(torch.rand_like(t)),
              reps=n_rep)
    eval_cond("Shuffle gates", gate_mode="shuffle", reps=n_rep)
    eval_cond("Random gates", gate_mode="random", reps=n_rep)
    eval_cond("Gates = 1", gate_mode="ones", reps=1)
    eval_cond("Gates = 0", gate_mode="zeros", reps=1)

    write_csv("metric3_prior_shuffling.csv",
              ["Condition", "Acc", "AccStd", "DeltaAcc", "FlipRate", "ConfDrop", "KL"],
              [[n, f"{a:.2f}", f"{s:.2f}", f"{a-base_acc:.2f}", f"{f:.2f}", f"{c:.6f}", f"{k:.6f}"]
               for n, a, s, f, c, k in results])

    names = [r[0] for r in results]
    flips = [r[3] for r in results]
    confs = [r[4] for r in results]
    colors = ["#4C72B0"] + ["#C44E52"] * (len(names) - 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    axes[0].barh(range(len(names)), flips, color=colors, edgecolor="k", lw=0.35)
    axes[0].set_yticks(range(len(names))); axes[0].set_yticklabels(names, fontsize=8)
    axes[0].invert_yaxis(); axes[0].set_xlabel("Flip rate (%)")
    axes[0].set_title("Prior Shuffling Controls (Prediction Flip)")
    axes[1].barh(range(len(names)), confs, color=colors, edgecolor="k", lw=0.35)
    axes[1].set_yticks(range(len(names))); axes[1].set_yticklabels(names, fontsize=8)
    axes[1].invert_yaxis(); axes[1].set_xlabel("Mean confidence drop")
    axes[1].set_title("Prior Shuffling Controls (Confidence)")
    fig.tight_layout()
    save_pdf(fig, "fig3_prior_shuffling")


# -------------------- 4) Intervention --------------------
def fig_intervention(model, cache):
    print("\n[4] Intervention")
    cls = cache["cls"]
    topic = cache["topic"]
    labels_np = cache["labels"]
    labels = torch.tensor(labels_np, device=DEVICE)
    K = topic.size(1)
    logits0, _ = predict_from_cache(model, cls, topic)
    base_acc = (logits0.argmax(1) == labels).float().mean().item() * 100
    mean_w = topic.mean(0).cpu().numpy()
    order = np.argsort(-mean_w)

    def eval_mask(mask):
        mask_t = torch.tensor(mask, dtype=torch.float32).view(1, -1)
        t2 = topic * mask_t
        t2 = t2 / (t2.sum(1, keepdim=True) + 1e-10)
        l1, _ = predict_from_cache(model, cls, t2)
        return effect(logits0, l1, labels)

    single = []
    for k in range(K):
        m = np.ones(K, np.float32); m[k] = 0
        acc, flip, conf, kl = eval_mask(m)
        single.append((k, acc, base_acc - acc, flip, conf, kl))
        print(f"  zero T{k}: flip {flip:.2f}%")

    write_csv("metric4_single_topic_ablation.csv",
              ["Topic", "Acc", "DropAcc", "FlipRate", "ConfDrop", "KL"],
              [[f"T{k}", f"{a:.2f}", f"{d:.2f}", f"{f:.2f}", f"{c:.6f}", f"{kl:.6f}"]
               for k, a, d, f, c, kl in single])

    cumulative = [(0, base_acc, 0.0, 0.0, 0.0)]
    m = np.ones(K, np.float32)
    for rank, k in enumerate(order, 1):
        m[k] = 0
        acc, flip, conf, kl = eval_mask(m)
        cumulative.append((rank, acc, base_acc - acc, flip, conf))

    write_csv("metric4_cumulative_remove.csv",
              ["NumRemoved", "Acc", "DropAcc", "FlipRate", "ConfDrop"],
              [[n, f"{a:.2f}", f"{d:.2f}", f"{f:.2f}", f"{c:.6f}"] for n, a, d, f, c in cumulative])

    keep_only = []
    for k_keep in [1, 3, 5, 8, 10, 15]:
        m = np.zeros(K, np.float32)
        for k in order[:k_keep]:
            m[k] = 1
        acc, flip, conf, kl = eval_mask(m)
        keep_only.append((k_keep, acc, base_acc - acc, flip, conf))

    class_rows = []
    for c in range(NUM_CLASSES):
        idx = np.where(labels_np == c)[0]
        if len(idx) == 0:
            continue
        top_k = int(topic[idx].mean(0).argmax())
        m = np.ones(K, np.float32); m[top_k] = 0
        mask_t = torch.tensor(m, dtype=torch.float32).view(1, -1)
        t2 = topic[idx] * mask_t
        t2 = t2 / (t2.sum(1, keepdim=True) + 1e-10)
        l0 = logits0[idx]
        l1, _ = predict_from_cache(model, cls[idx], t2)
        y = labels[idx]
        acc, flip, conf, kl = effect(l0, l1, y)
        class_rows.append((c, top_k, flip, conf))
        print(f"  {CLASS_NAMES[c]}: zero T{top_k} flip {flip:.2f}%")

    write_csv("metric4_classwise_intervention.csv",
              ["Class", "ZeroedTopic", "FlipRate", "ConfDrop"],
              [[CLASS_NAMES[c], f"T{tk}", f"{f:.2f}", f"{cd:.6f}"] for c, tk, f, cd in class_rows])

    plot_items = sorted(single, key=lambda x: -x[3])
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    axes[0, 0].bar(range(K), [x[3] for x in plot_items], color="#C44E52", edgecolor="k", lw=0.3)
    axes[0, 0].set_xticks(range(K))
    axes[0, 0].set_xticklabels([rf"$T_{{{x[0]}}}$" for x in plot_items], fontsize=8)
    axes[0, 0].set_ylabel("Flip rate (%)")
    axes[0, 0].set_title("Single-Topic Intervention")

    axes[0, 1].plot([c[0] for c in cumulative], [c[3] for c in cumulative], "o-", color="#4C72B0", lw=2)
    ax2 = axes[0, 1].twinx()
    ax2.plot([c[0] for c in cumulative], [c[4] for c in cumulative], "s--", color="#55A868", lw=2)
    axes[0, 1].set_xlabel("Removed topics (by mean weight)")
    axes[0, 1].set_ylabel("Flip rate (%)", color="#4C72B0")
    ax2.set_ylabel("Conf drop", color="#55A868")
    axes[0, 1].set_title("Cumulative Topic Removal")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot([k for k, *_ in keep_only], [f for _, _, _, f, _ in keep_only], "s-", color="#55A868", lw=2, label="Flip")
    axes[1, 0].plot([k for k, *_ in keep_only], [c for *_, c in keep_only], "o--", color="#C44E52", lw=2, label="Conf")
    axes[1, 0].set_xlabel("Keep top-$k$ topics")
    axes[1, 0].set_title("Retain Top-$k$ Topics")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].barh(range(len(class_rows)), [f for *_, f, __ in class_rows], color="#DD8452", edgecolor="k", lw=0.3)
    axes[1, 1].set_yticks(range(len(class_rows)))
    axes[1, 1].set_yticklabels([CLASS_NAMES[c] for c, *_ in class_rows], fontsize=8)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("Flip rate (%)")
    axes[1, 1].set_title("Zero Class-Dominant Topic")
    fig.tight_layout()
    save_pdf(fig, "fig4_intervention")


# -------------------- 5) Gating --------------------
def fig_gating(model, cache):
    print("\n[5] Gating attribution")
    cls = cache["cls"].to(DEVICE)
    topic = cache["topic"].to(DEVICE)
    gates = cache["gate"].numpy()
    labels = cache["labels"]
    N, H = gates.shape

    # Grad x Gate in batches
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
    gate_attr = np.concatenate(attr_list, 0)
    mean_gate = gates.mean(0)
    mean_attr = gate_attr.mean(0)
    importance = np.argsort(-mean_attr)

    write_csv("metric5_gate_importance.csv",
              ["GateDim", "MeanGate", "GradxAct"],
              [[h, f"{mean_gate[h]:.6f}", f"{mean_attr[h]:.6e}"] for h in range(H)])

    labels_t = torch.tensor(labels, device=DEVICE)
    logits0, _ = predict_from_cache(model, cache["cls"], cache["topic"])

    fracs = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    abl = []
    for fr in fracs:
        keep = np.ones(H, np.float32)
        n = int(round(fr * H))
        keep[importance[:n]] = 0
        l1, _ = predict_from_cache(model, cache["cls"], cache["topic"], gate_keep=torch.tensor(keep))
        acc, flip, conf, kl = effect(logits0, l1, labels_t)
        abl.append((fr, acc, flip, conf, kl))
        print(f"  zero top {int(fr*100)}% gates: flip {flip:.2f}%")

    write_csv("metric5_gate_ablation.csv",
              ["FracZeroed", "Acc", "FlipRate", "ConfDrop", "KL"],
              [[f"{fr:.1f}", f"{a:.2f}", f"{f:.2f}", f"{c:.6f}", f"{kl:.6f}"]
               for fr, a, f, c, kl in abl])

    rng = np.random.RandomState(0)
    rand_flips = []
    for fr in fracs:
        vals = []
        for _ in range(2):
            keep = np.ones(H, np.float32)
            n = int(round(fr * H))
            if n > 0:
                keep[rng.choice(H, n, replace=False)] = 0
            l1, _ = predict_from_cache(model, cache["cls"], cache["topic"], gate_keep=torch.tensor(keep))
            _, flip, _, _ = effect(logits0, l1, labels_t)
            vals.append(flip)
        rand_flips.append((fr, float(np.mean(vals)), float(np.std(vals))))

    top_n = min(40, H)
    top_idx = importance[:top_n]
    topics_np = cache["topic"].numpy()
    tsel = np.argsort(-topics_np.mean(0))[:15]
    gsel = importance[:32]
    mat = np.zeros((len(tsel), len(gsel)))
    for i, k in enumerate(tsel):
        for j, h in enumerate(gsel):
            if topics_np[:, k].std() < 1e-12 or gates[:, h].std() < 1e-12:
                mat[i, j] = 0
            else:
                mat[i, j] = np.corrcoef(topics_np[:, k], gates[:, h])[0, 1]

    fig = plt.figure(figsize=(12.5, 9.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar(range(top_n), mean_attr[top_idx], color="#4C72B0", edgecolor="k", lw=0.2)
    ax0.set_ylabel(r"$|$Grad $\times$ Gate$|$")
    ax0.set_title(f"Gating Attribution (Top-{top_n})")
    ax0.set_xticks(range(top_n))
    ax0.set_xticklabels([str(i) for i in top_idx], fontsize=5.5, rotation=90)

    ax1 = fig.add_subplot(gs[0, 1])
    im = ax1.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax1.set_yticks(range(len(tsel))); ax1.set_yticklabels([rf"$T_{{{k}}}$" for k in tsel], fontsize=8)
    ax1.set_xticks(range(len(gsel))); ax1.set_xticklabels([str(i) for i in gsel], fontsize=6, rotation=90)
    ax1.set_title("Topic–Gate Correlation")
    # annotate like fig1 topic–scattering heatmap
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if abs(v) >= 0.12:
                ax1.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.2,
                         color=("white" if abs(v) > 0.55 else "black"))
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot([x[0] for x in abl], [x[2] for x in abl], "o-", color="#C44E52", lw=2, label="Attr-guided")
    ax2.errorbar([x[0] for x in rand_flips], [x[1] for x in rand_flips],
                 yerr=[x[2] for x in rand_flips], fmt="s--", color="#4C72B0", lw=2, capsize=3, label="Random")
    ax2.set_xlabel("Fraction of gate dims zeroed")
    ax2.set_ylabel("Flip rate (%)")
    ax2.set_title("Gating Ablation (Flip)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    class_gate = np.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        m = labels == c
        if m.any():
            class_gate[c] = gates[m].mean()
    ax3.bar(range(NUM_CLASSES), class_gate, color="#8172B3", edgecolor="k", lw=0.3)
    ax3.set_xticks(range(NUM_CLASSES))
    ax3.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=8)
    ax3.set_ylabel("Mean gate activation")
    ax3.set_title("Class-wise Mean Gating Strength")

    fig.tight_layout()
    save_pdf(fig, "fig5_gating_attribution")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    model, pgcl_path, fusion_path = load_models()
    t_items, r_items = load_paired(T_VAL, RSD_VAL)
    print("samples:", len(t_items))
    cache = extract_cache(model, t_items, r_items)
    acc = 100.0 * (cache["preds"] == cache["labels"]).mean()
    print(f"Fusion Acc: {acc:.2f}%")
    print("native topic active:", int((cache["topic"].numpy().std(0) > 1e-8).sum()), "/15")
    print("softmax topic active:", int((cache["topic_soft"].numpy().std(0) > 1e-8).sum()), "/15")

    fig_mapping(cache)
    print("Correlation matrix written to", SAVE_DIR)
    print("Done:", SAVE_DIR)


if __name__ == "__main__":
    main()
