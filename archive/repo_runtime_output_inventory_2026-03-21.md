# Repo Runtime And Output Inventory (No Data)

## 1. Scope

This inventory excludes `data/` and focuses on what is needed to run the project and what the repository already outputs.

## 2. Runtime-Critical Files

### 2.1 Core Code

- `src/A1_sweep_fp32_img224_bs16_512.py`
- `src/A1_sweep_fp16_img224_bs16_512.py`
- `src/A2_e2e_photos.py`
- `src/A2_e2e_photos_opt.py`
- `src/A2_final_opt.py`
- `src/A2_final_opt_v2.py`
- `src/A2_final_opt_v3.py`
- `src/run_a2_grid_sweep.py`
- `src/run_a2_grid_sweep_V2.py`
- `src/run_a2_grid_sweep_V3.py`
- `src/test_uni_npu.py`
- `src/A3/` entire directory

### 2.2 Dependency And Docs

- `src/A3/requirements.txt`
- `src/A3/README.md`
- `README.md`
- `README-process.md`
- `.gitignore`

### 2.3 Runtime Weights

Required for the UNI main line:

- `assets/ckpts/UNI/pytorch_model.bin`
- `assets/ckpts/UNI/config.json`
- `assets/ckpts/UNI/uni.jpg`

Optional experimental weight:

- `assets/ckpts/hf/hub/model.safetensors`

## 3. Existing Output Artifacts

### 3.1 A1 Outputs

- `logs/A1_fp16__img224__bs16-512__iters120_warm30_r2.tsv`
- `logs/A1_fp32__img224__bs16-512__iters120_warm30_r2.tsv`

### 3.2 A2 Outputs

- `logs/A2_final_baseline.json`
- `logs/A2_final_baseline.tsv`
- `logs/A2_fp16_test_bs96.tsv`
- `logs/A2_gmix_bs96.tsv`
- `logs/A2_sweep_summary.tsv`
- `logs/A2final_fp16_test_bs32.tsv`
- `logs/A2opt_fp16_test_bs32.tsv`
- `logs/sweep_v3/`

### 3.3 A3 Outputs

- `logs/A3_output/A_phase/` data checks and manifests
- `logs/A3_output/B_phase/` Photos Stage1 training outputs
- `logs/A3_output/C_phase/` WSI manifest and Stage1.5 outputs
- `logs/A3_output/D_phase/` train/test WSI feature bags
- `logs/A3_output/E_phase/` MIL / TileAgg / thresholds / CV summaries
- `logs/A3_output/F_phase/` attention heatmaps
- `logs/A3_output/G_phase/` final inference CSVs
- `logs/A3_output/reports/` summary reports

## 4. Current Best A3 Delivery Files

- `logs/A3_output/B_phase/stage1_uni_large_cv5_ms512_1024_1536_v1/cv_summary_mean_std.csv`
- `logs/A3_output/C_phase/stage1p5_uni_large_L1_s448_mt40_random_v1/best.pt`
- `logs/A3_output/D_phase/wsi_test_features_L1_s448_uniStage1p5_v1/`
- `logs/A3_output/E_phase/tileagg_thresholds_L1_s448_uniStage1p5_v1.json`
- `logs/A3_output/G_phase/tileagg_test_wsi_pred_L1_s448_uniStage1p5_topk16_v1.csv`
- `logs/A3_output/reports/final_wsi_pipeline_summary.json`

## 5. What Is Required On A Fresh Server

If you want to rerun from scratch:

- entire `src/`
- `assets/ckpts/UNI/`
- root docs and config files
- your own `data/BACH/`
- Python deps from `src/A3/requirements.txt`
- environment-level `torch_npu`
- system-level OpenSlide shared library

If you want to reuse the already trained A3 best line:

- everything above
- selected `logs/A3_output/` artifacts listed in section 4

## 6. Size Notes

Large files currently present locally:

- `assets/ckpts/UNI/pytorch_model.bin` ~ 1.21 GB
- several `logs/A3_output/.../best.pt` ~ 1.21 GB
- `assets/ckpts/hf/hub/model.safetensors` ~ 346 MB
- top-level size summary:
  - `assets/` ~ 1.5 GB
  - `logs/` ~ 18 GB
  - `archive/` ~ 204 KB
  - `src/` ~ 876 KB

## 7. Raw Manifest

See `archive/repo_file_manifest_no_data_2026-03-21.txt` for the full file list.
