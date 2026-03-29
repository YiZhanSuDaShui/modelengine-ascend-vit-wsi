import os, time, statistics
import torch
import torch_npu  # noqa
import timm

CKPT = "./assets/ckpts/UNI/pytorch_model.bin"
assert os.path.exists(CKPT), f"missing ckpt: {CKPT}"

IMG_SIZE = 224
BATCHES = [16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512]

ITERS = 120
WARMUP = 30
REPEATS = 2

def build_model(device):
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=IMG_SIZE,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    state = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model

def run_once(model, x):
    with torch.autocast(device_type="npu", dtype=torch.float16):
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
    for _ in range(REPEATS):
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
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    model = build_model(device)

    os.makedirs("logs", exist_ok=True)
    out_path = f"logs/fp16_sweep_{IMG_SIZE}_bs16_512.csv"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("img_size,precision,batch,iters/warmup,repeats,ms_per_patch,patch_per_s\n")
        for bs in BATCHES:
            ms, ips = safe_bench(model, bs)
            print(f"{IMG_SIZE},{'FP16(AMP)'},{bs},{ITERS}/{WARMUP},{REPEATS},{ms},{ips}")
            f.write(f"{IMG_SIZE},FP16(AMP),{bs},{ITERS}/{WARMUP},{REPEATS},{ms},{ips}\n")

    print(f"\n[Saved] {out_path}")
    print("Tip: sweet spot = 最大 patch/s 或最小 ms_per_patch（且不 OOM）")

if __name__ == "__main__":
    main()