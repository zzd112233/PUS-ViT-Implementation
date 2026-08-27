"""
联合训练脚本：
1. 在RSD数据集上使用PGCL架构训练VisionTransformer，得到主题模型与物理可引导卷积层
2. 在T数据集上训练VisionTransformer（不使用PGCL），并将预训练的PGCL模块作为指导信号
"""

import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from VisionTransformer_7_RSD import VisionTransformer as ViT_RSD
from VisionTransformer_7_T import VisionTransformer as ViT_T


class TxtDataset(Dataset):
    def __init__(self, txt_path):
        samples = []
        with open(txt_path, "r") as fh:
            for line in fh:
                items = line.strip().split()
                if len(items) >= 2:
                    samples.append((items[0], int(items[1])))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index + 1:]
    return torch.cat((arr1, arr2), dim=0)


def calc_loss(outputs, labels, device):
    outputs = outputs.to(device)
    labels = labels.to(device)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    return loss.mean()


def build_dataloader(txt_path, batch_size, shuffle, num_workers=0):
    dataset = TxtDataset(txt_path)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train_rsd_with_pgcl(config):
    """
    在RSD数据集上训练PGCL模型，返回训练好的模型以及其主题/PGCL模块
    """
    device = config["device"]
    use_pgcl = True

    # 先检查实际数据的通道数，避免硬编码的通道数不匹配
    train_loader_temp = build_dataloader(config["train_txt"], batch_size=1, shuffle=False)
    actual_in_channels = None
    for inputs_list, _ in train_loader_temp:
        if len(inputs_list) > 0:
            sample_tensor = torch.load(inputs_list[0]).to(device)
            if len(sample_tensor.shape) == 4:  # (B, C, H, W)
                actual_in_channels = sample_tensor.shape[1]
            elif len(sample_tensor.shape) == 3:  # (C, H, W)
                actual_in_channels = sample_tensor.shape[0]
            break
    
    if actual_in_channels is None:
        raise RuntimeError("无法从训练数据中获取通道数，请检查数据文件格式")
    
    print(f"检测到实际数据通道数: {actual_in_channels} (配置中的 rsd_in_channels: {config.get('rsd_in_channels', '未指定')})")
    
    # 默认启用双分支架构
    use_dual_branch = config.get("use_dual_branch", True)
    
    model = ViT_RSD(
        embed_dim=64,
        num_classes=config["num_classes"],
        num_topics=config["num_topics"],
        use_pgcl=use_pgcl,
        depth=config.get("rsd_depth", 3),
        in_channels=actual_in_channels,  # 使用实际数据的通道数
        use_dual_branch=use_dual_branch  # 启用双分支架构
    ).to(device)
    
    if use_dual_branch:
        print("已启用双分支RSD架构：")
        print("  - 分支1：RSD数据直接与SupervisedTopicModel结合")
        print("  - 分支2：RSD数据通过ViT encoder提取高阶特征，然后与主题模型结合")
        print("  - 融合：两个分支的增强特征进行融合")

    # 冻结PhysicsGuidedConvLayer的参数，不参与训练
    if use_dual_branch and hasattr(model, 'fusion_pgcl'):
        # 双分支架构：只冻结 PhysicsGuidedConvLayer 的参数（branch1_pgcl 和 branch2_pgcl）
        # 不冻结 topic_model 和 fusion_layer 等其他参数
        if hasattr(model.fusion_pgcl, 'branch1_pgcl'):
            for p in model.fusion_pgcl.branch1_pgcl.parameters():
                p.requires_grad = False
        if hasattr(model.fusion_pgcl, 'branch2_pgcl'):
            for p in model.fusion_pgcl.branch2_pgcl.parameters():
                p.requires_grad = False
        print("✓ PhysicsGuidedConvLayer (branch1_pgcl 和 branch2_pgcl) 参数已冻结，不参与训练")
        print("  注意: topic_model 和 fusion_layer 等其他参数仍可训练")
    elif hasattr(model, 'pgcl'):
        # 单分支架构：冻结 pgcl 的所有参数
        for p in model.pgcl.parameters():
            p.requires_grad = False
        print("✓ PhysicsGuidedConvLayer (pgcl) 参数已冻结，不参与训练")

    # 只优化需要训练的参数（排除PGCL参数）
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    
    optimizer = torch.optim.AdamW(trainable_params, lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config["epochs"])
    
    # 打印可训练参数数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in trainable_params)
    frozen_params_count = total_params - trainable_params_count
    print(f"  总参数数: {total_params:,}, 可训练参数: {trainable_params_count:,}, 冻结参数: {frozen_params_count:,}")

    train_loader = build_dataloader(config["train_txt"], config["batch_size"], shuffle=True)
    test_loader = build_dataloader(config["test_txt"], config["batch_size"], shuffle=False)

    best_acc = 0
    best_path = None
    os.makedirs(config["save_dir"], exist_ok=True)

    print("=" * 60)
    print("Stage-1: 训练 RSD PGCL 模型")
    print("=" * 60)

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        total_cls_loss = 0
        total_topic_loss = 0
        count = 0

        for inputs_list, labels in train_loader:
            labels = labels.to(device)
            # 构建batch tensor，根据第一个样本的实际通道数自适应，不再依赖rsd_in_channels
            batch_tensor = None
            for sample_path in inputs_list:
                sample_tensor = torch.load(sample_path).to(device)  # 形状应为 (B, C, 7, 7) 或 (1, C, 7, 7)
                if batch_tensor is None:
                    batch_tensor = sample_tensor
                else:
                    batch_tensor = torch.cat((batch_tensor, sample_tensor), dim=0)

            optimizer.zero_grad()
            logits, cls_features, topic_representation = model(batch_tensor)
            cls_loss = calc_loss(logits, labels, device)
            
            # 如果使用双分支架构，计算两个分支的主题损失
            if hasattr(model, 'use_dual_branch') and model.use_dual_branch:
                # 获取分支1的特征（通过融合PGCL层的投影层）
                branch1_features = model.fusion_pgcl.branch1_projection(batch_tensor)
                branch1_topic_loss = model.fusion_pgcl.branch1_topic_model.compute_supervised_loss(branch1_features, labels)
                
                # 分支2的主题损失（使用CLS特征）
                branch2_topic_loss = model.fusion_pgcl.branch2_topic_model.compute_supervised_loss(cls_features, labels)
                
                # 组合两个分支的主题损失
                topic_loss = branch1_topic_loss + branch2_topic_loss
            else:
                # 单分支模式
                topic_loss = model.topic_model.compute_supervised_loss(cls_features, labels)
            
            total_loss_batch = cls_loss + config["topic_lambda"] * topic_loss
            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()
            total_cls_loss += cls_loss.item()
            total_topic_loss += topic_loss.item()
            count += 1

        avg_loss = total_loss / count
        print(f"[RSD][Epoch {epoch+1}] Loss: {avg_loss:.4f} (Cls: {total_cls_loss/count:.4f}, Topic: {total_topic_loss/count:.4f})")

        scheduler.step()

        # 验证
        acc = evaluate(model, test_loader, device, use_pgcl=True)
        if acc >= best_acc:
            best_acc = acc
            # 按验证精度命名当前最佳模型
            best_path = os.path.join(
                config["save_dir"],
                f"pgcl_model_best_{best_acc:.2f}.pth"
            )
            torch.save(model.state_dict(), best_path)
            print(f"  保存最佳PGCL模型: {best_path} (Acc: {best_acc:.2f}%)")

    # 加载最佳模型
    if best_path is not None:
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"RSD PGCL Stage 完成, 最佳Acc: {best_acc:.2f}% (路径: {best_path})")
    else:
        raise RuntimeError("RSD PGCL 训练阶段未能保存任何模型，请检查训练/验证流程。")

    # 如果使用双分支架构，返回融合PGCL层
    if hasattr(model, 'use_dual_branch') and model.use_dual_branch:
        guidance_fusion_pgcl = copy.deepcopy(model.fusion_pgcl).eval()
        for p in guidance_fusion_pgcl.parameters():
            p.requires_grad = False
        # 为了向后兼容，也返回分支2的topic_model和pgcl
        guidance_topic = copy.deepcopy(model.fusion_pgcl.branch2_topic_model).eval()
        guidance_pgcl = copy.deepcopy(model.fusion_pgcl.branch2_pgcl).eval()
        for p in guidance_topic.parameters():
            p.requires_grad = False
        for p in guidance_pgcl.parameters():
            p.requires_grad = False
        return guidance_topic, guidance_pgcl, best_path, guidance_fusion_pgcl
    else:
        # 单分支模式
        guidance_topic = copy.deepcopy(model.topic_model).eval()
        guidance_pgcl = copy.deepcopy(model.pgcl).eval()
        for p in guidance_topic.parameters():
            p.requires_grad = False
        for p in guidance_pgcl.parameters():
            p.requires_grad = False
        return guidance_topic, guidance_pgcl, best_path, None


def evaluate(model, data_loader, device, use_pgcl, in_channels=None):
    """
    评估 RSD 模型

    Args:
        in_channels: 已弃用，仅为兼容旧接口，实际通道数从数据本身推断
    """
    model.eval()
    total_preds = []
    total_labels = []

    with torch.no_grad():
        for inputs_list, labels in data_loader:
            labels = labels.to(device)
            batch_tensor = None
            for sample_path in inputs_list:
                sample_tensor = torch.load(sample_path).to(device)
                if batch_tensor is None:
                    batch_tensor = sample_tensor
                else:
                    batch_tensor = torch.cat((batch_tensor, sample_tensor), dim=0)

            outputs = model(batch_tensor)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            total_preds.append(preds)
            total_labels.append(labels.cpu().numpy())

    preds = np.concatenate(total_preds)
    labels = np.concatenate(total_labels)
    acc = round((preds == labels).sum() / len(labels) * 100, 2)
    print(f"  Eval Acc: {acc:.2f}%")
    return acc


class GuidedVisionTransformerT(nn.Module):
    """
    将预训练的PGCL模块用于T数据集的VisionTransformer
    """
    def __init__(self, base_model, guidance_topic, guidance_pgcl, num_classes, fusion_dim=128):
        super().__init__()
        self.backbone = base_model  # VisionTransformer_7_T (use_pgcl=False)
        self.guidance_topic = guidance_topic
        self.guidance_pgcl = guidance_pgcl
        self.num_classes = num_classes

        hidden_dim = fusion_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(self.backbone.embed_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        logits_base, cls_features = self.backbone(x)  # (B, num_classes), (B, embed_dim)

        with torch.no_grad():
            topic_representation, _ = self.guidance_topic(cls_features)
            guided_features = self.guidance_pgcl(cls_features, topic_representation)  # (B, hidden_dim)

        fused_feature = torch.cat([cls_features, guided_features], dim=1)
        fusion_logits = self.fusion_head(fused_feature)

        return fusion_logits, logits_base


def train_t_with_guidance(config, guidance_topic, guidance_pgcl, rsd_model_path):
    device = config["device"]

    base_model = ViT_T(
        embed_dim=64,
        num_classes=config["num_classes"],
        num_topics=config["num_topics"],
        use_pgcl=False,
        rsd_model_path=rsd_model_path
    ).to(device)

    guided_model = GuidedVisionTransformerT(
        base_model,
        guidance_topic.to(device),
        guidance_pgcl.to(device),
        num_classes=config["num_classes"]
    ).to(device)

    # 冻结物理引导模块的参数
    for p in guided_model.guidance_topic.parameters():
        p.requires_grad = False
    for p in guided_model.guidance_pgcl.parameters():
        p.requires_grad = False
    print("✓ 物理引导模块（guidance_topic 和 guidance_pgcl）参数已冻结")

    # 只优化需要训练的参数（backbone 和 fusion_head）
    trainable_params = list(guided_model.backbone.parameters()) + list(guided_model.fusion_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config["epochs"])

    train_loader = build_dataloader(config["train_txt"], config["batch_size"], shuffle=True)
    test_loader = build_dataloader(config["test_txt"], config["batch_size"], shuffle=False)

    best_acc = 0
    best_path = None
    os.makedirs(config["save_dir"], exist_ok=True)

    print("=" * 60)
    print("Stage-2: 训练 T 数据集 (融合PGCL指导)")
    print("=" * 60)

    for epoch in range(config["epochs"]):
        guided_model.train()
        total_loss = 0
        count = 0

        for inputs_list, labels in train_loader:
            labels = labels.to(device)
            batch_tensor = None
            for sample_path in inputs_list:
                sample_tensor = torch.load(sample_path).to(device)
                if batch_tensor is None:
                    batch_tensor = sample_tensor
                else:
                    batch_tensor = torch.cat((batch_tensor, sample_tensor), dim=0)

            optimizer.zero_grad()
            fusion_logits, logits_base = guided_model(batch_tensor)

            # 交叉熵损失 + KD损失
            ce_loss = calc_loss(fusion_logits, labels, device)
            kd_loss = torch.nn.functional.kl_div(
                torch.log_softmax(fusion_logits, dim=1),
                torch.softmax(logits_base.detach(), dim=1),
                reduction="batchmean"
            )
            total_batch_loss = ce_loss + config["kd_lambda"] * kd_loss
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()
            count += 1

        avg_loss = total_loss / count
        print(f"[T][Epoch {epoch+1}] Loss: {avg_loss:.4f}")

        scheduler.step()

        acc = evaluate_fusion(guided_model, test_loader, device, config["t_in_channels"])
        if acc >= best_acc:
            best_acc = acc
            # 按验证精度命名当前最佳融合模型
            best_path = os.path.join(
                config["save_dir"],
                f"fusion_model_best_{best_acc:.2f}.pth"
            )
            torch.save(guided_model.state_dict(), best_path)
            print(f"  保存最佳Fusion模型: {best_path} (Acc: {best_acc:.2f}%)")

    if best_path is None:
        raise RuntimeError("T 阶段训练未能保存任何 Fusion 模型，请检查训练/验证流程。")

    print(f"T Stage 完成, 最佳Acc: {best_acc:.2f}% (路径: {best_path})")
    return best_path


def evaluate_fusion(model, data_loader, device, in_channels):
    model.eval()
    total_preds = []
    total_labels = []

    with torch.no_grad():
        for inputs_list, labels in data_loader:
            labels = labels.to(device)
            batch_tensor = None
            for sample_path in inputs_list:
                sample_tensor = torch.load(sample_path).to(device)
                if batch_tensor is None:
                    batch_tensor = sample_tensor
                else:
                    batch_tensor = torch.cat((batch_tensor, sample_tensor), dim=0)

            fusion_logits, _ = model(batch_tensor)
            preds = torch.argmax(fusion_logits, dim=1).cpu().numpy()
            total_preds.append(preds)
            total_labels.append(labels.cpu().numpy())

    preds = np.concatenate(total_preds)
    labels = np.concatenate(total_labels)
    acc = round((preds == labels).sum() / len(labels) * 100, 2)
    print(f"  Fusion Eval Acc: {acc:.2f}%")
    return acc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rsd_config = {
        "device": device,
        "train_txt": "G:\\eXplaniableCNN\\0PUS_ViT\\Barnaul\\RSD\\Train.txt",
        "test_txt": "G:\\eXplaniableCNN\\0PUS_ViT\\Barnaul\\RSD\\Test.txt",
        "batch_size": 32,
        "epochs": 50,
        "lr": 4e-4,
        # 将类别数从7改为9，用于9类地物分类
        "num_classes": 9,
        "num_topics": 15,
        "rsd_depth": 3,
        "topic_lambda": 0.1,
        "rsd_in_channels": 12,
        "save_dir": "G:\\eXplaniableCNN\\1Hyperparameter_discussion\\3Generalization_analysis_of_PGCL\\freezen\\model"
    }

    t_config = {
        "device": device,
        "train_txt": "G:\\eXplaniableCNN\\0PUS_ViT\\Barnaul\\T\\Train.txt",
        "test_txt": "G:\\eXplaniableCNN\\0PUS_ViT\\Barnaul\\T\\Test.txt",
        "batch_size": 32,
        "epochs": 50,
        "lr": 4e-4,
        # 将类别数从7改为9，用于9类地物分类
        "num_classes": 9,
        "num_topics": 15,
        "kd_lambda": 0.2,
        "t_in_channels": 10,
        "save_dir": "G:\\eXplaniableCNN\\1Hyperparameter_discussion\\3Generalization_analysis_of_PGCL\\freezen\\model"
    }

    result = train_rsd_with_pgcl(rsd_config)
    if len(result) == 4:
        guidance_topic, guidance_pgcl, rsd_best_path, guidance_fusion_pgcl = result
    else:
        # 向后兼容
        guidance_topic, guidance_pgcl, rsd_best_path = result
        guidance_fusion_pgcl = None
    train_t_with_guidance(t_config, guidance_topic, guidance_pgcl, rsd_best_path)

