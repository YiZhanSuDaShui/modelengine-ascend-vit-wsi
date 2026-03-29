# A1 纯推理基线测试脚本 - FP16 Sweep
# 用于测量不同 batch size 下 ViT-Large 的推理性能
# 用途：作为后续所有优化（AIPP、ATC、INT8 等）的"统一裁判"基准

import os, time, statistics, datetime  # os:路径操作, time:计时, statistics:中位数, datetime:时间戳
import torch  # PyTorch 核心库
import torch_npu  # noqa  # 华为 NPU 加速库（必须导入才能启用 NPU）
import timm  # Torch Image Models - 预定义模型库

# ==================== 硬编码配置参数 ====================
CKPT = "./assets/ckpts/UNI/pytorch_model.bin"  # 预训练模型权重路径
IMG_SIZE = 224  # 输入图像尺寸（ViT 标准 224x224）
BATCHES = [16,24,32,48,64,96,128,160,192,224,256,320,384,448,512]  # 要 sweep 的 batch size 列表
ITERS = 120  # 正式计时时，每个 batch size 跑多少个 iteration
WARMUP = 30  # 正式计时前，先跑多少次推理来预热（让缓存、算子编译等达到稳态）
REPEATS = 2  # 重复跑几次，然后取中位数（抗抖动）

# 输出文件名：根据参数自动生成，方便追溯
OUT_TSV = f"logs/A1_fp16__img{IMG_SIZE}__bs16-512__iters{ITERS}_warm{WARMUP}_r{REPEATS}.tsv"

# ==================== 工具函数 ====================

# 获取当前时间字符串，用于日志打印
# 返回格式：HH:MM:SS 例如 "14:30:05"
def now(): return datetime.datetime.now().strftime("%H:%M:%S")


# ==================== 模型构建 ====================

# 构建并加载 ViT-Large-Patch16-224 模型
# 参数:
#   device: torch.device, 模型要加载到的设备（npu:0）
# 返回:
#   model: 加载好权重并.eval()后的模型
def build_model(device):
    # 使用 timm 库创建模型
    # vit_large_patch16_224: ViT-Large, 16x16 patch, 224 输入
    # init_values=1e-5: 初始化缩放因子
    # num_classes=0: 去掉分类头，只保留特征提取器（backbone）
    # dynamic_img_size=True: 允许非标准图像尺寸输入
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=IMG_SIZE, patch_size=16,
        init_values=1e-5, num_classes=0,
        dynamic_img_size=True,
    )
    # 从磁盘加载预训练权重到 CPU 内存
    state = torch.load(CKPT, map_location="cpu")
    # 加载权重到模型（strict=True 确保权重完全匹配）
    model.load_state_dict(state, strict=True)
    # 切换到推理模式，并移动到指定设备
    model.eval().to(device)
    return model

# ==================== 推理运行 ====================

# 执行一次模型推理（单次 forward）
# 参数:
#   model: 已加载的模型
#   x: 输入 tensor (bs, 3, IMG_SIZE, IMG_SIZE)
# 返回:
#   模型输出
def run_once(model, x):
    # torch.autocast: 自动混合精度（AMP）
    # device_type="npu": 指定 NPU 设备
    # dtype=torch.float16: FP16 推理
    # 相当于自动做 float32 -> float16 计算 -> float32 输出
    with torch.autocast(device_type="npu", dtype=torch.float16):
        return model(x)

# ==================== 基准测试核心 ====================

# 对单个 batch size 进行一次完整的性能测试
# 参数:
#   model: 已加载的模型
#   bs: batch size
# 返回:
#   ms_per_img: 每张图像的延迟（毫秒）
#   ips: 吞吐量（图像/秒）
def bench_one(model, bs):
    # 获取模型所在的设备
    device = next(model.parameters()).device
    
    # 创建随机输入 tensor
    # 形状: (bs, 3, IMG_SIZE, IMG_SIZE)
    # 数据: 标准正态分布随机值
    x = torch.randn(bs, 3, IMG_SIZE, IMG_SIZE, device=device)

    # ==================== Warmup 阶段 ====================
    # torch.inference_mode(): 推理模式，比 no_grad 更轻量（不创建反向图）
    with torch.inference_mode():
        # 跑 WARMUP 次推理，让 NPU 缓存、编译达到稳态
        for _ in range(WARMUP):
            _ = run_once(model, x)
        # 强制同步，确保 warmup 全部完成后再开始计时
        # NPU 是异步执行的，不同步会计时不准
        torch.npu.synchronize()

    # ==================== 正式计时阶段 ====================
    # 记录开始时间
    t0 = time.time()
    
    # 跑 ITERS 次推理
    with torch.inference_mode():
        for _ in range(ITERS):
            _ = run_once(model, x)
    
    # 强制同步，确保所有推理完成
    torch.npu.synchronize()
    
    # 记录结束时间
    t1 = time.time()

    # ==================== 计算指标 ====================
    # 总耗时（秒）
    total = t1 - t0
    
    # 总共处理了多少张图像 = iters * batch_size
    imgs = ITERS * bs
    
    # 每张图像的延迟（毫秒）
    # total / imgs 得到秒/图像，再 * 1000 得到毫秒/图像
    ms_per_img = (total / imgs) * 1000.0
    
    # 吞吐量：图像/秒 = 总图像数 / 总耗时
    ips = imgs / total
    
    return ms_per_img, ips

# ==================== 鲁棒测试包装 ====================

# 安全的基准测试包装器
# 作用：重复多次测试，取中位数；捕获 OOM 异常
# 参数:
#   model: 已加载的模型
#   bs: batch size
# 返回:
#   ms: 延迟中位数（毫秒）或 "OOM"
#   ips: 吞吐量中位数（图像/秒）或 "OOM"
def safe_bench(model, bs):
    ms_list, ips_list = [], []  # 存储每次 repeat 的结果
    
    # 循环 REPEATS 次
    for r in range(1, REPEATS + 1):
        # 打印进度日志
        print(f"[{now()}] FP16 bs={bs} repeat {r}/{REPEATS} ...", flush=True)
        
        try:
            # 执行一次基准测试
            ms, ips = bench_one(model, bs)
            ms_list.append(ms)
            ips_list.append(ips)
        except Exception as e:
            # 捕获异常，检查是否是 OOM
            msg = str(e).lower()
            if "out of memory" in msg or "oom" in msg:
                # 如果是 OOM，返回特殊标记
                return "OOM", "OOM"
            # 其他异常重新抛出
            raise
    
    # 返回多次测试的中位数（抗抖动）
    return statistics.median(ms_list), statistics.median(ips_list)

# ==================== 主函数 ====================

# 主入口：执行完整的 sweep 测试
def main():
    # 检查模型权重文件是否存在
    assert os.path.exists(CKPT), f"missing ckpt: {CKPT}"
    
    # 创建输出目录（如果不存在）
    os.makedirs("logs", exist_ok=True)
    
    # 初始化 NPU 设备
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    
    # 打印初始化信息
    print(f"[{now()}] Build model FP16 on {device}, img={IMG_SIZE} ...", flush=True)
    
    # 构建并加载模型
    model = build_model(device)
    
    # TSV 文件表头
    # 各列含义：图像尺寸 | 精度 | BatchSize | iters/warmup | repeats | 延迟(ms/patch) | 吞吐量(patch/s) | 备注
    header = "ImgSize(H×W)\tPrecision\tBatchSize\titers/warmup\trepeats\tLatency(ms/patch)\tThroughput(patch/s)\tNotes"
    
    # 打开输出文件，准备写入
    with open(OUT_TSV, "w", encoding="utf-8") as f:
        # 写入表头
        f.write(header + "\n")
        
        # 拼接 iters/warmup 字符串，如 "120/30"
        itw = f"{ITERS}/{WARMUP}"
        
        # 遍历每个 batch size 进行测试
        for bs in BATCHES:
            # 打印当前测试开始信息
            print(f"\n[{now()}] === Start FP16 img={IMG_SIZE} bs={bs} ===", flush=True)
            
            # 执行安全测试
            ms, ips = safe_bench(model, bs)
            
            # 构建备注字段
            note = "OOM" if ms == "OOM" else ""
            
            # 拼接一行数据（Tab 分隔）
            line = f"{IMG_SIZE}×{IMG_SIZE}\tFP16(AMP)\t{bs}\t{itw}\t{REPEATS}\t{ms}\t{ips}\t{note}"
            
            # 写入文件并 flush 落盘
            f.write(line + "\n")
            f.flush()
            
            # 打印完成信息
            print(f"[{now()}] === Done  FP16 bs={bs}  ms={ms}  ips={ips} ===", flush=True)
    
    # 打印最终输出路径
    print(f"\n[{now()}] Saved ONLY output: {OUT_TSV}", flush=True)

# ==================== 程序入口 ====================

# Python 脚本标准入口
# 只有直接运行此脚本时才会执行 main()
# 如果被 import 则不会执行
if __name__ == "__main__":
    main()