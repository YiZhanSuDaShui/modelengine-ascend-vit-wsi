import os, time, argparse, statistics
import torch
import torch_npu  # noqa: F401
import timm

CKPT = "./assets/ckpts/UNI/pytorch_model.bin"

def build_model(device, img_size: int):
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=img_size,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    state = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model

def run_once(model, x, amp: bool):
    if amp:
        with torch.autocast(device_type="npu", dtype=torch.float16):
            return model(x)
    return model(x)

def bench_one(model, bs: int, img_size: int, iters: int, warmup: int, amp: bool):
    device = next(model.parameters()).device
    # 随机输入：等价于 bs 张 (3,img_size,img_size) 的图片张量
    x = torch.randn(bs, 3, img_size, img_size, device=device)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = run_once(model, x, amp=amp)
        torch.npu.synchronize()

    t0 = time.time()
    with torch.inference_mode():
        for _ in range(iters):
            _ = run_once(model, x, amp=amp)
        torch.npu.synchronize()
    t1 = time.time()

    total = t1 - t0
    imgs = iters * bs
    ms_per_img = (total / imgs) * 1000.0
    ips = imgs / total
    return ms_per_img, ips

def safe_bench(model, bs, img_size, iters, warmup, amp, repeats):
    ms_list, ips_list = [], []
    for _ in range(repeats):
        try:
            ms, ips = bench_one(model, bs, img_size, iters, warmup, amp)
            ms_list.append(ms)
            ips_list.append(ips)
        except Exception as e:
            msg = str(e).lower()
            if "out of memory" in msg or "oom" in msg:
                return "OOM", "OOM"
            # 其他异常直接抛出，让你看到真实报错
            raise
    # 用中位数更抗抖动
    return statistics.median(ms_list), statistics.median(ips_list)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_sizes", type=str, default="224", help="e.g. 224 or 224,384")
    ap.add_argument("--batches", type=str, default="1,2,4,8,16", help="e.g. 1,2,4,8,16,32")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    assert os.path.exists(CKPT), f"missing ckpt: {CKPT}"

    device = torch.device("npu:0")
    torch.npu.set_device(device)

    img_sizes = [int(x) for x in args.img_sizes.split(",") if x.strip()]
    batches = [int(x) for x in args.batches.split(",") if x.strip()]

    os.makedirs("logs", exist_ok=True)

    # 输出两份：CSV（方便计算） + Word表（制表符分隔）
    print("img_size,precision,batch,iters/warmup,latency_ms_per_patch,throughput_patch_per_s")
    word_lines = []
    word_lines.append("ImgSize(H×W)\tPrecision\tBatchSize\titers/warmup\tLatency(ms/patch)\tThroughput(patch/s)\tS_same_bs(vs FP32)\tNotes")

    for img in img_sizes:
        # 每个 img_size 建一次模型（结构相同，输入尺寸不同）
        model = build_model(device, img)

        results = {"FP32": {}, "FP16": {}}

        for prec, amp in [("FP32", False), ("FP16(AMP)", True)]:
            for bs in batches:
                ms, ips = safe_bench(model, bs, img, args.iters, args.warmup, amp, args.repeats)
                # 打印CSV行
                print(f"{img},{prec},{bs},{args.iters}/{args.warmup},{ms},{ips}")

                results["FP32" if prec=="FP32" else "FP16"][bs] = (ms, ips)

        # 生成 Word 表（同 batch 加速比）
        for bs in batches:
            ms32, ips32 = results["FP32"].get(bs, ("", ""))
            ms16, ips16 = results["FP16"].get(bs, ("", ""))
            # speedup 同 batch：FP32_ms / FP16_ms
            speed = ""
            if isinstance(ms32, (int, float)) and isinstance(ms16, (int, float)):
                speed = f"{(ms32/ms16):.2f}x"
            note = ""
            if ms32 == "OOM" or ms16 == "OOM":
                note = "OOM（该batch不可用）"
            word_lines.append(
                f"{img}×{img}\tFP32\t{bs}\t{args.iters}/{args.warmup}\t{ms32}\t{ips32}\t1.00\t{note}"
            )
            word_lines.append(
                f"{img}×{img}\tFP16(AMP)\t{bs}\t{args.iters}/{args.warmup}\t{ms16}\t{ips16}\t{speed}\t{note}"
            )
        word_lines.append("---\t---\t---\t---\t---\t---\t---\t---")

    # 写出 Word 可直接粘贴的制表符表
    out_path = "logs/tableA_word.tsv"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(word_lines) + "\n")

    print(f"\n[Saved] Word-ready table -> {out_path}")

if __name__ == "__main__":
    main()