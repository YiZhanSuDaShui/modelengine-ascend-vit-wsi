import os                         # 导入操作系统接口模块，用于文件路径等操作
import torch                      # 导入 PyTorch 深度学习框架
import torch_npu                  # 导入华为 NPU 后端支持，确保 NPU 可用
import timm                       # 导入 timm（PyTorch Image Models）库，用于加载预训练视觉模型
from PIL import Image             # 导入 Python Imaging Library，用于图像读取与处理
from torchvision import transforms  # 导入 torchvision 的图像预处理工具

CKPT_DIR = "./assets/ckpts/UNI"  # 定义模型检查点所在目录
CKPT = os.path.join(CKPT_DIR, "pytorch_model.bin")  # 拼接出模型权重文件完整路径
IMG = os.path.join(CKPT_DIR, "uni.jpg")  # 拼接出示例图片完整路径

assert os.path.exists(CKPT), f"missing: {CKPT}"  # 断言检查权重文件是否存在，若不存在则抛出错误提示

device = torch.device("npu:0")   # 指定使用第 0 号 NPU 设备
torch.npu.set_device(device)     # 设置当前进程默认使用的 NPU 设备

model = timm.create_model(
    "vit_large_patch16_224",     # 选择 ViT-Large 模型，输入 224×224，patch 大小 16×16
    img_size=224,                # 输入图像尺寸
    patch_size=16,               # 每个 patch 的像素大小
    init_values=1e-5,            # LayerScale 初始值
    num_classes=0,               # 分类头输出维度为 0，即只取特征提取 backbone
    dynamic_img_size=True,       # 允许动态输入尺寸（虽然此处固定 224）
)

state = torch.load(CKPT, map_location="cpu")  # 将权重加载到 CPU 内存
model.load_state_dict(state, strict=True)     # 严格匹配键值，加载权重到模型
model.eval().to(device)                       # 切换为评估模式，并将模型移至 NPU

transform = transforms.Compose([                # 定义图像预处理流程
    transforms.Resize(256),                     # 将短边缩放到 256
    transforms.CenterCrop(224),                 # 中心裁剪出 224×224
    transforms.ToTensor(),                       # 转为 Tensor，数值范围 [0,1]
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # ImageNet 标准化
])

if os.path.exists(IMG):                         # 若示例图片存在
    img = Image.open(IMG).convert("RGB")       # 读取并转为 RGB 格式
else:                                           # 否则
    img = Image.new("RGB", (224, 224), (128, 128, 128))  # 创建一张灰色占位图

x = transform(img).unsqueeze(0).to(device)      # 预处理并增加 batch 维度，再移至 NPU
print("input tensor shape:", tuple(x.shape))   # 打印输入张量形状

with torch.inference_mode():                    # 禁用梯度计算，加速推理
    feat = model(x)                             # 前向传播，提取特征

print("device:", device)                        # 打印当前使用的设备
print("feature shape:", tuple(feat.shape))     # 打印输出特征张量形状
print("feature sample:", feat[0, :8].detach().cpu())  # 打印前 8 个特征值（移至 CPU 后）
