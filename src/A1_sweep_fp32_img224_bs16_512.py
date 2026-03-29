import os, time, statistics, datetime
import torch
import torch_npu  # noqa
import timm

CKPT = "./assets/ckpts/UNI/pytorch_model.bin"
IMG_SIZE = 224
BATCHES = [16,24,32,48,64,96,128,160,192,224,256,320,384,448,512]
ITERS = 120
WARMUP = 30
REPEATS = 2

OUT_TSV = f"logs/A1_fp32__img{IMG_SIZE}__bs16-512__iters{ITERS}_warm{WARMUP}_r{REPEATS}.tsv"

def now(): return datetime.datetime.now().strftime("%H:%M:%S")

def build_model(device):
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=IMG_SIZE, patch_size=16,
        init_values=1e-5, num_classes=0,
        dynamic_img_size=True,
    )
    state = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model

def run_once(model, x):
    return model(x)

def bench_one(model, bs):
    device = next(model.parameters()).device
    x = torch.randn(bs, 3, IMG_SIZE, IMG_SIZE, device=device)

    with torch.inference_mode():
        for _ in range(WARMUP):
            _ = run_once(model, x)
        torch.npu.synchronize()

    t0 = time.time()
    with torch.inference_mode():
        for _ in range(ITERS):
            _ = run_once(model, x)
        torch.npu.synchronize()
    t1 = time.time()

    total = t1 - t0
    imgs = ITERS * bs
    ms_per_img = (total / imgs) * 1000.0
    ips = imgs / total
    return ms_per_img, ips

def safe_bench(model, bs):
    ms_list, ips_list = [], []
    for r in range(1, REPEATS + 1):
        print(f"[{now()}] FP32 bs={bs} repeat {r}/{REPEATS} ...", flush=True)
        try:
            ms, ips = bench_one(model, bs)
            ms_list.append(ms)
            ips_list.append(ips)
        except Exception as e:
            msg = str(e).lower()
            if "out of memory" in msg or "oom" in msg:
                return "OOM", "OOM"
            raise
    return statistics.median(ms_list), statistics.median(ips_list)

def main():
    assert os.path.exists(CKPT), f"missing ckpt: {CKPT}"
    os.makedirs("logs", exist_ok=True)

    device = torch.device("npu:0")
    torch.npu.set_device(device)

    print(f"[{now()}] Build model FP32 on {device}, img={IMG_SIZE} ...", flush=True)
    model = build_model(device)

    header = "ImgSize(H×W)\tPrecision\tBatchSize\titers/warmup\trepeats\tLatency(ms/patch)\tThroughput(patch/s)\tNotes"
    with open(OUT_TSV, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        itw = f"{ITERS}/{WARMUP}"

        for bs in BATCHES:
            print(f"\n[{now()}] === Start FP32 img={IMG_SIZE} bs={bs} ===", flush=True)
            ms, ips = safe_bench(model, bs)
            note = "OOM" if ms == "OOM" else ""
            line = f"{IMG_SIZE}×{IMG_SIZE}\tFP32\t{bs}\t{itw}\t{REPEATS}\t{ms}\t{ips}\t{note}"
            f.write(line + "\n")
            f.flush()
            print(f"[{now()}] === Done  FP32 bs={bs}  ms={ms}  ips={ips} ===", flush=True)

    print(f"\n[{now()}] Saved ONLY output: {OUT_TSV}", flush=True)

if __name__ == "__main__":
    main()