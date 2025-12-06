import { FileNode, MetricData, ArchitectureLayer, WorkflowStage, EngineInsight } from './types';

// --- Simulated Performance Data based on Document Targets ---
export const PERFORMANCE_DATA: MetricData[] = [
  { name: 'CPU / PyTorch (Baseline)', throughput: 15, latency: 450, accuracy: 0.885 },
  { name: 'Ascend 910B (FP16)', throughput: 280, latency: 25, accuracy: 0.884 },
  { name: 'Ascend 910B (INT8 + Opt)', throughput: 420, latency: 14, accuracy: 0.881 },
];

export const LAYERS: ArchitectureLayer[] = [
  {
    title: "Application Layer",
    description: "End-user interfaces for pathologists and researchers",
    items: ["Doctor Workstation", "HarmonyOS App", "RESTful API Gateway"],
    color: "bg-blue-100 border-blue-300 text-blue-800"
  },
  {
    title: "ModelEngine Service Layer",
    description: "Cloud orchestration, inference scheduling, and monitoring",
    items: ["ModelEngine Core", "Inference Scheduler", "Log & Monitor", "Job Queue"],
    color: "bg-indigo-100 border-indigo-300 text-indigo-800"
  },
  {
    title: "Infrastructure & Hardware Layer",
    description: "Physical compute resources and storage",
    items: ["Ascend 910B NPU", "Kunpeng Servers", "CANN 7.0", "Distributed Storage"],
    color: "bg-slate-100 border-slate-300 text-slate-800"
  }
];

export const WORKFLOW_STAGES: WorkflowStage[] = [
  {
    key: 'connect',
    titleKey: 'stage.connect',
    descKey: 'stage.connect.desc',
    status: 'ready',
    eta: '3s'
  },
  {
    key: 'aipp',
    titleKey: 'stage.aipp',
    descKey: 'stage.aipp.desc',
    status: 'pending',
    eta: '6s'
  },
  {
    key: 'atc',
    titleKey: 'stage.atc',
    descKey: 'stage.atc.desc',
    status: 'pending',
    eta: '8s'
  },
  {
    key: 'amct',
    titleKey: 'stage.amct',
    descKey: 'stage.amct.desc',
    status: 'pending',
    eta: '25s'
  },
  {
    key: 'engine',
    titleKey: 'stage.engine',
    descKey: 'stage.engine.desc',
    status: 'pending',
    eta: '12s'
  },
  {
    key: 'monitor',
    titleKey: 'stage.monitor',
    descKey: 'stage.monitor.desc',
    status: 'pending',
    eta: 'real-time'
  }
];

export const ENGINE_INSIGHTS: EngineInsight[] = [
  {
    engine: 'ATC + AIPP',
    mode: 'Static Shape + Preprocess Offload',
    speedup: '3.1x',
    timeTradeoff: '-6% flexibility (shape fixed)',
    description: 'Compile-time fusion and hardware AIPP remove host preprocessing and reduce DDR trips.'
  },
  {
    engine: 'AMCT INT8 Hybrid',
    mode: 'Sparse-aware Mixed Precision',
    speedup: '3.6x',
    timeTradeoff: '-0.8% accuracy / +12 min calibration',
    description: 'Attention kept at FP16 while FFN runs INT8; calibration introduces setup overhead but preserves recall.'
  },
  {
    engine: 'ModelEngine Runtime',
    mode: 'Pipeline + Profiling',
    speedup: '3.0x',
    timeTradeoff: '-3% scheduling flexibility',
    description: 'Batch streaming with profiling traces keeps the 910B saturated while reporting live telemetry.'
  }
];

// --- ACTUAL CODE IMPLEMENTATIONS AS REQUESTED ---

const CODE_WSI_PREPROCESS = `import openslide
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def extract_foreground_mask(slide, level=2, threshold=240):
    """Extract tissue foreground using Otsu thresholding."""
    img = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]))
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    # Otsu thresholding to separate tissue (darker) from background (lighter)
    _, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask

def patch_generator(slide_path, patch_size=256, level=0):
    """Generate patches from foreground regions."""
    slide = openslide.OpenSlide(str(slide_path))
    mask = extract_foreground_mask(slide)
    
    # Scale factor between mask level and extraction level
    scale = slide.level_downsamples[2] / slide.level_downsamples[level]
    
    w, h = slide.level_dimensions[level]
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            # Check mask at corresponding coordinates
            mask_y, mask_x = int(y // scale), int(x // scale)
            if mask[mask_y, mask_x] > 0: # If tissue exists
                patch = slide.read_region((x, y), level, (patch_size, patch_size))
                yield patch.convert('RGB'), (x, y)

def process_batch(wsi_dir, output_dir):
    """Batch processing wrapper."""
    files = list(Path(wsi_dir).glob("*.svs"))
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(lambda f: patch_generator(f), files)
`;

const CODE_UNI_BACKBONE = `import torch
import torch.nn as nn
import timm

class UNIBackbone(nn.Module):
    """
    Wrapper for the UNI ViT-Large model.
    Loads pretrained weights from Mahmood Lab's assets.
    """
    def __init__(self, model_path, num_classes=4, freeze_backbone=True):
        super(UNIBackbone, self).__init__()
        # Load ViT-Large-Patch16-224
        self.backbone = timm.create_model(
            'vit_large_patch16_224', 
            pretrained=False, 
            num_classes=0 # Remove default head
        )
        
        # Load UNI specific weights
        print(f"Loading UNI weights from {model_path}...")
        state_dict = torch.load(model_path, map_location='cpu')
        self.backbone.load_state_dict(state_dict, strict=False)
        
        if freeze_backbone:
            forparam in self.backbone.parameters():
                param.requires_grad = False
                
        # Downstream classification head (Linear Probe)
        self.head = nn.Linear(1024, num_classes) # ViT-Large embed_dim=1024

    def forward(self, x):
        # x shape: [B, 3, 224, 224]
        features = self.backbone(x)
        logits = self.head(features)
        return logits, features

# Dummy entry for testing
if __name__ == "__main__":
    model = UNIBackbone('./checkpoints/uni.pth')
    dummy_input = torch.randn(1, 3, 224, 224)
    logits, _ = model(dummy_input)
    print("Output shape:", logits.shape)
`;

const CODE_EXPORT_ONNX = `import torch
from uni_backbone import UNIBackbone

def export_to_onnx(model_path, output_path, batch_size=32):
    model = UNIBackbone(model_path, freeze_backbone=True)
    model.eval()
    
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    # Export options
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14, # Higher opset for ViT support
        do_constant_folding=True,
        input_names=['input'],
        output_names=['logits', 'features'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'features': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {output_path}")

if __name__ == "__main__":
    export_to_onnx('./checkpoints/uni_finetuned.pth', './models/uni_v1.onnx')
`;

const CODE_ATC_SCRIPT = `#!/bin/bash
# ATC Conversion Script for Ascend 910B

MODEL_NAME="uni_v1"
INPUT_ONNX="./models/\${MODEL_NAME}.onnx"
OUTPUT_OM="./models/\${MODEL_NAME}_bs32"
SOC_VERSION="Ascend910B"

# Set environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

echo "Starting ATC conversion..."

atc --model=$INPUT_ONNX \
    --framework=5 \
    --output=$OUTPUT_OM \
    --soc_version=$SOC_VERSION \
    --input_shape="input:32,3,224,224" \
    --insert_op_conf=./ascend_tools/aipp.cfg \
    --enable_small_channel=1 \
    --log=error \
    --op_select_implmode=high_performance

if [ $? -eq 0 ]; then
    echo "ATC Conversion Success: \${OUTPUT_OM}.om generated."
else
    echo "ATC Conversion Failed!"
    exit 1
fi
`;

const CODE_AIPP_CFG = `aipp_op {
    aipp_mode: static
    input_format: YUV420SP_U8
    csc_switch: true
    var_reci_switch: true
    
    # Color Space Conversion (YUV to RGB)
    matrix_r0c0: 298
    matrix_r0c1: 0
    matrix_r0c2: 409
    matrix_r1c0: 298
    matrix_r1c1: -100
    matrix_r1c2: -208
    matrix_r2c0: 298
    matrix_r2c1: 516
    matrix_r2c2: 0
    
    # Input bias (subtract mean) - UNI Model specific mean
    mean_chn_0: 123
    mean_chn_1: 117
    mean_chn_2: 104
    
    # Reciprocal of variance (1/std)
    var_reci_chn_0: 0.01712
    var_reci_chn_1: 0.01750
    var_reci_chn_2: 0.01742
}
`;

const CODE_ASCEND_INFER = `import acl
import numpy as np
import time

class AscendInference:
    def __init__(self, model_path, device_id=0):
        self.device_id = device_id
        self.model_path = model_path
        self._init_resource()

    def _init_resource(self):
        acl.init()
        acl.rt.set_device(self.device_id)
        self.context, _ = acl.rt.create_context(self.device_id)
        self.model_id, _ = acl.mdl.load_from_file(self.model_path)
        self.model_desc = acl.mdl.create_desc()
        acl.mdl.get_desc(self.model_desc, self.model_id)

    def infer(self, input_data):
        # 1. Prepare Dataset (Host -> Device)
        input_dataset, input_buffers = self._prepare_input(input_data)
        output_dataset, output_buffers = self._prepare_output()
        
        # 2. Execute Inference
        start = time.time()
        acl.mdl.execute(self.model_id, input_dataset, output_dataset)
        infer_time = time.time() - start
        
        # 3. Retrieve results (Device -> Host)
        results = self._process_output(output_buffers)
        
        # Cleanup buffers
        self._release_dataset(input_dataset, input_buffers)
        self._release_dataset(output_dataset, output_buffers)
        
        return results, infer_time

    def _prepare_input(self, input_np):
        # Implementation of creating aclDataBuffer and copying memory
        # ... (Omitted for brevity, standard ACL boilerplate) ...
        pass
        
    def release(self):
        acl.mdl.unload(self.model_id)
        acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()

# Usage
if __name__ == "__main__":
    net = AscendInference("./models/uni_v1_bs32.om")
    dummy = np.random.rand(32, 3, 224, 224).astype(np.float32)
    res, latency = net.infer(dummy)
    print(f"Batch(32) Inference time: {latency*1000:.2f} ms")
`;

// --- Directory Structure ---
export const PROJECT_STRUCTURE: FileNode = {
  name: 'project_root',
  type: 'folder',
  children: [
    {
      name: 'configs',
      type: 'folder',
      children: [
        { name: 'uni_finetune.yaml', type: 'file', language: 'yaml' },
        { name: 'datasets.yaml', type: 'file', language: 'yaml' }
      ]
    },
    {
      name: 'data_utils',
      type: 'folder',
      children: [
        { name: 'wsi_preprocess.py', type: 'file', language: 'python', content: CODE_WSI_PREPROCESS },
        { name: 'color_normalize.py', type: 'file', language: 'python' },
        { name: 'bach_dataset.py', type: 'file', language: 'python' }
      ]
    },
    {
      name: 'models',
      type: 'folder',
      children: [
        { name: 'uni_backbone.py', type: 'file', language: 'python', content: CODE_UNI_BACKBONE },
        { name: 'mil_heads.py', type: 'file', language: 'python' },
        { name: 'export_onnx.py', type: 'file', language: 'python', content: CODE_EXPORT_ONNX }
      ]
    },
    {
      name: 'ascend_tools',
      type: 'folder',
      children: [
        { name: 'atc_convert.sh', type: 'file', language: 'bash', content: CODE_ATC_SCRIPT },
        { name: 'amct_config.json', type: 'file', language: 'json' },
        { name: 'quantization.py', type: 'file', language: 'python' },
        { name: 'aipp.cfg', type: 'file', language: 'properties', content: CODE_AIPP_CFG }
      ]
    },
    {
      name: 'inference',
      type: 'folder',
      children: [
        { name: 'ascend_infer.py', type: 'file', language: 'python', content: CODE_ASCEND_INFER },
        { name: 'postprocess.py', type: 'file', language: 'python' },
        { name: 'benchmark.py', type: 'file', language: 'python' }
      ]
    },
    {
      name: 'scripts',
      type: 'folder',
      children: [
        { name: 'run_train_pytorch.sh', type: 'file', language: 'bash' },
        { name: 'run_export_onnx.sh', type: 'file', language: 'bash' },
        { name: 'run_convert_om.sh', type: 'file', language: 'bash' },
        { name: 'run_infer_ascend.sh', type: 'file', language: 'bash' }
      ]
    }
  ]
};