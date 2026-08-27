import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

import torch, math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import loadmat, savemat
from scipy import ndimage
import h5py
# from VGG16model import vgg16
# from ResNets import  resnet50
from time import *
# from main import alexnet
# from AlexNet import AlexNet
# from VGG import VGG19
import copy
from VisionTransformer_7_T import VisionTransformer as ViT_T
from VisionTransformer_7_RSD import VisionTransformer as ViT_RSD
from Train_Fusion import GuidedVisionTransformerT

begin_time = time()
# python调用visdom：python -m visdom.server

"""# **2.数据处理部分**"""


def addZeroPadding(X, margin=2):  #
    newX = np.zeros((
        X.shape[0] + 2 * margin,
        X.shape[1] + 2 * margin,
        X.shape[2]
    ))
    newX[margin:X.shape[0] + margin, margin:X.shape[1] + margin, :] = X
    return newX


"""# **3.读取数据建立dataloader**"""
windowSize = 7  # 测试数据集这一部分，窗口大小应该也给改一下，像margin等值是不是也得动一下？？？？？？？？？,这里可以改为5、7、9大小的滑动窗口，64的有些大了
num_classes = 9


def smooth_segmentation_by_connectivity(label_map, num_classes, min_region_size=90, neighbor_radius=9):
    """
    基于空间连通性与邻域多数类的后处理：
    1）对每个类别做连通域分解；
    2）将面积小于 min_region_size 的“小斑块”像素，重赋值为其邻域内的多数类别，
       以此减少零碎地物、增强整体连通性。

    Args:
        label_map: 2D numpy array，分类结果（值为0或1..num_classes）
        num_classes: 类别数
        min_region_size: 判定“小斑块”的最小像素数阈值
        neighbor_radius: 邻域半径（3表示7x7窗口）
    """
    h, w = label_map.shape
    label_map = label_map.astype(np.int32).copy()

    for cls in range(1, num_classes + 1):
        mask = (label_map == cls)
        if not mask.any():
            continue

        # 连通域标记
        labeled, num_comp = ndimage.label(mask)
        if num_comp == 0:
            continue

        sizes = np.bincount(labeled.ravel())
        # 选择“小斑块”ID（跳过0号背景）
        small_ids = [i for i, s in enumerate(sizes) if i != 0 and s < min_region_size]
        if not small_ids:
            continue

        for comp_id in small_ids:
            ys, xs = np.where(labeled == comp_id)
            for y, x in zip(ys, xs):
                y0 = max(0, y - neighbor_radius)
                y1 = min(h, y + neighbor_radius + 1)
                x0 = max(0, x - neighbor_radius)
                x1 = min(w, x + neighbor_radius + 1)

                window = label_map[y0:y1, x0:x1]
                # 去掉背景与自身小斑块标签，以邻域“其它地物”的多数类作为重映射目标
                window_flat = window.reshape(-1)
                window_flat = window_flat[window_flat != 0]
                if window_flat.size == 0:
                    continue

                new_label = np.bincount(window_flat).argmax()
                label_map[y, x] = new_label

    return label_map

# 构建融合模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 与 Train_Fusion.py 中保持一致的模型目录（num_topics = 15）
model_root = 'G:\\eXplaniableCNN\\1Hyperparameter_discussion\\3Generalization_analysis_of_PGCL\\freezen\\model'

# 自动查找最新的模型文件
def find_latest_model(model_dir, prefix):
    """查找指定前缀的最新模型文件"""
    if not os.path.exists(model_dir):
        return None
    model_files = [f for f in os.listdir(model_dir) if f.startswith(prefix) and f.endswith('.pth')]
    if not model_files:
        return None
    # 按修改时间排序，返回最新的
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    return os.path.join(model_dir, model_files[0])

# 指定精度的模型路径（可根据需要修改精度值）
# 如果指定文件不存在，尝试自动查找最新模型
pgcl_model_path = os.path.join(model_root, 'pgcl_model_best_94.00.pth')
fusion_model_path = os.path.join(model_root, 'fusion_model_best_99.90.pth')

# 如果指定文件不存在，尝试查找最新模型
if not os.path.exists(pgcl_model_path):
    print(f"警告: 指定的RSD PGCL模型未找到: {pgcl_model_path}")
    pgcl_model_path = find_latest_model(model_root, 'pgcl_model_best_')
    if pgcl_model_path:
        print(f"自动找到最新的RSD PGCL模型: {pgcl_model_path}")
    else:
        raise FileNotFoundError(f"RSD PGCL 模型未找到，请检查模型目录: {model_root}")

if not os.path.exists(fusion_model_path):
    print(f"警告: 指定的Fusion模型未找到: {fusion_model_path}")
    fusion_model_path = find_latest_model(model_root, 'fusion_model_best_')
    if fusion_model_path:
        print(f"自动找到最新的Fusion模型: {fusion_model_path}")
    else:
        raise FileNotFoundError(f"Fusion 模型未找到，请检查模型目录: {model_root}")

# 从模型路径中提取精度值，用于生成结果文件名
def extract_acc_from_path(path):
    """从模型路径中提取精度值"""
    base = os.path.basename(path)
    try:
        # 形如 fusion_model_best_99.97.pth
        if 'fusion_model_best_' in base:
            acc_str = base.replace('fusion_model_best_', '').replace('.pth', '')
        elif 'pgcl_model_best_' in base:
            acc_str = base.replace('pgcl_model_best_', '').replace('.pth', '')
        else:
            acc_str = 'unknown'
        return acc_str
    except Exception:
        return 'unknown'

# 提取融合模型的精度值（用于结果文件名）
fusion_acc = extract_acc_from_path(fusion_model_path)

print("=" * 60)
print("加载 RSD PGCL 模型，用于提供物理引导...")
print("=" * 60)
# 检查是否使用双分支架构
rsd_model = ViT_RSD(
    embed_dim=64,
    num_classes=num_classes,
    num_topics=15,
    use_pgcl=True,
    use_dual_branch=True  # 启用双分支架构
).to(device)
rsd_model.load_state_dict(torch.load(pgcl_model_path, map_location=device))
rsd_model.eval()

# 检查是否使用双分支架构
if hasattr(rsd_model, 'use_dual_branch') and rsd_model.use_dual_branch:
    # 双分支架构：使用融合PGCL层
    print("检测到双分支架构，使用融合PGCL层...")
    guidance_topic = copy.deepcopy(rsd_model.fusion_pgcl.branch2_topic_model).to(device).eval()
    guidance_pgcl = copy.deepcopy(rsd_model.fusion_pgcl.branch2_pgcl).to(device).eval()
else:
    # 单分支架构：使用原始topic_model和pgcl
    print("使用单分支架构...")
    guidance_topic = copy.deepcopy(rsd_model.topic_model).to(device).eval()
    guidance_pgcl = copy.deepcopy(rsd_model.pgcl).to(device).eval()

for p in guidance_topic.parameters():
    p.requires_grad = False
for p in guidance_pgcl.parameters():
    p.requires_grad = False
print("✓ 物理引导模块（guidance_topic 和 guidance_pgcl）参数已冻结")

print("加载融合模型 (ViT_T + PGCL指导)...")
base_model = ViT_T(
    embed_dim=64,
    num_classes=num_classes,
    num_topics=15,
    use_pgcl=False,
    rsd_model_path=pgcl_model_path  # 与训练阶段保持一致，包含 rsd_pgcl_layers 结构
).to(device)

fusion_model = GuidedVisionTransformerT(
    base_model,
    guidance_topic,
    guidance_pgcl,
    num_classes=num_classes
).to(device)

# 加载融合模型权重；允许存在多余键（例如 rsd_pgcl_layers 中的权重）
fusion_state = torch.load(fusion_model_path, map_location=device)
fusion_model.load_state_dict(fusion_state, strict=False)
fusion_model.eval()
print("✓ 融合模型加载成功，开始推理")
"""# **6.可视化分类结果**"""
T_test = h5py.File('G:\\eXplaniableCNN\\0PUS_ViT\\Barnaul\\T\\Barnaul_T_Norm.mat', 'r')
T1 = T_test['Train9'][:]
mask_test = h5py.File('E:\\1工作\\1张帅影\\Data_An\\真值图\\地物标签\\Barnual地物标签选择\\9类\\Barnaul_9_gt.mat', 'r')
data_gt = mask_test['cdata'][:]
data_gt = torch.from_numpy(data_gt.transpose(1, 0))
# 这里不对预处理好的数据进行最大最小值线性归一化了，直接在matlab中对数据进行非线性归一化处理
# data_hsi=minmax_normalize(T1)
data_hsi1 = T1
print(np.shape(data_hsi1))
data_hsi1 = torch.from_numpy(data_hsi1.transpose(2, 1, 0))
height1, width1, c1 = data_hsi1.shape
# data_gt = data_testgt  # sio.loadmat(os.path.join(data_path, 'mask_test.mat'))['mask_test']
margin = (windowSize - 1) // 2  # 邻域
data_hsi1 = addZeroPadding(data_hsi1, margin=margin)
# 这里不要再对数据进行线性归一化了！已经在matlab中处理好了
# data_hsi=minmax_normalize(data_hsi)
# 逐像素预测类别
outputs = np.zeros((height1, width1))

print("=" * 60)
print("开始对极化SAR数据进行分类（ViT_T + PGCL指导）...")
print(f"图像尺寸: {height1} × {width1}")
print(f"窗口大小: {windowSize} × {windowSize}")
print("=" * 60)

with torch.no_grad():
    for i in range(height1):
        if (i + 1) % 100 == 0:
            print(f"处理进度: {i + 1}/{height1} 行 ({100 * (i + 1) / height1:.1f}%)")

        for j in range(width1):

            # 这里将测试集的真值图也放进来，考虑选取部分点进行分类，并且拿这部分点进行混淆矩阵的评测，按照这种方法去做，精度就能提高！
            if int(data_gt[i, j]) != 0:  # 背景值为0的不进行分类，这个主要目的，先测试标注图像的位置能否则正确分类，如果效果还不错，那么就注释掉这一句话，进而对整幅图像进行分类，或者对别的数据进行分类
                image_patch = data_hsi1[i:i + windowSize, j:j + windowSize, :]
                image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2])
                X_test_image = torch.FloatTensor(image_patch.transpose(0, 3, 1, 2)).to(device)

            # 模型前向传播（融合PGCL知识）
                fusion_logits, _ = fusion_model(X_test_image)
                prediction = torch.argmax(fusion_logits, dim=1).item()
                outputs[i][j] = prediction + 1  # 按原逻辑保留 +1
# 保存分类结果（原始）
output_dir = model_root  # 与训练阶段保存模型的目录保持一致
os.makedirs(output_dir, exist_ok=True)
raw_output_path = os.path.join(output_dir, f'result_full_raw_acc2_fusion_model_best_{fusion_acc}.mat')
sio.savemat(raw_output_path, {'output': outputs})
print(f'\n原始分类结果已保存到: {raw_output_path}')

# 基于距离/邻域相似度的后处理：聚合零碎地物，提高连通性
print("\n开始进行基于连通性与邻域多数类的后处理...")
outputs_smoothed = smooth_segmentation_by_connectivity(outputs, num_classes=num_classes,
                                                       min_region_size=90, neighbor_radius=9)

post_output_path = os.path.join(output_dir, f'result_full_post_acc_fusion_model_best_{fusion_acc}.mat')
sio.savemat(post_output_path, {'output': outputs_smoothed})
print(f'后处理后的分类结果已保存到: {post_output_path}')

# 计算统计信息（基于后处理结果）
total_pixels = np.sum(data_gt.numpy() != 0)
classified_pixels = np.sum(outputs_smoothed != 0)
print("\n" + "=" * 60)
print("分类统计信息:")
print("=" * 60)
print(f"总像素数: {total_pixels}")
print(f"已分类像素数: {classified_pixels}")
print(f"分类覆盖率: {100 * classified_pixels / total_pixels:.2f}%" if total_pixels > 0 else "N/A")
print(f"使用架构: ViT_T + PGCL指导 (融合模型)")
print("=" * 60)

print('\nALL Finish!!')
end_time = time()
run_time = end_time - begin_time
print(f'Running time: {run_time / 3600:.2f} hours ({run_time:.2f} seconds)')