import React, { createContext, useContext, useState, ReactNode } from 'react';

export type Language = 'en' | 'zh';

interface I18nContextType {
    language: Language;
    setLanguage: (lang: Language) => void;
    t: (key: string) => string;
}

const translations: Record<Language, Record<string, string>> = {
    en: {
        // Sidebar
        'sidebar.title': 'ModelEngine',
        'sidebar.subtitle': 'UNI ViT Accelerator',
        'sidebar.overview': 'Overview & Metrics',
        'sidebar.architecture': 'System Architecture',
        'sidebar.code': 'Project Repository',
        'sidebar.benchmark': 'Benchmark Results',
        'sidebar.deviceStatus': 'Device Status',
        'sidebar.npuLoad': 'NPU Load',

        // Dashboard
        'dashboard.title': 'Project Overview',
        'dashboard.description': 'Accelerating UNI Pathological Foundation Model (ViT) Inference on Ascend NPU. Focusing on WSI Patch classification for Breast Cancer (BACH dataset) using CANN, ATC, and AMCT technologies.',
        'dashboard.throughput': 'Throughput (Ascend INT8)',
        'dashboard.throughputSub': '+2700% vs CPU Baseline',
        'dashboard.latency': 'Avg WSI Latency',
        'dashboard.latencySub': 'Reduced from ~15 mins',
        'dashboard.accuracy': 'Model Accuracy (F1)',
        'dashboard.accuracySub': '-0.4% Loss (vs FP32)',
        'dashboard.dataset': 'Dataset (BACH)',
        'dashboard.datasetSub': 'Normal, Benign, InSitu, Invasive',
        'dashboard.throughputChart': 'Throughput Comparison (Patch/s)',
        'dashboard.latencyChart': 'Inference Latency (ms/batch)',
        'dashboard.datasetTitle': 'BACH Dataset Distribution',
        'dashboard.patchCount': 'Patch Count',
        'dashboard.description2': 'Description',
        'dashboard.controlPanel': 'One-stop control panel',
        'dashboard.npuStatus': 'Ascend 910B availability',
        'dashboard.checkButton': 'Check availability',
        'dashboard.monitoring': 'Real-time telemetry',
        'dashboard.currentEngine': 'Current accelerator',
        'dashboard.engineInsights': 'Accelerator insights & trade-offs',
        'dashboard.pipeline': 'Workflow tracker',
        'dashboard.tradeoff': 'Time flexibility loss',
        'dashboard.speedup': 'Speedup vs baseline',
        'dashboard.liveLoad': 'Live load',
        'dashboard.temperature': 'Temperature',
        'dashboard.stagesLabel': 'Pipeline stages',
        'dashboard.ready': 'Ready',
        'dashboard.running': 'Running',
        'dashboard.pending': 'Pending',
        'dashboard.blocked': 'Blocked',
        'dashboard.done': 'Done',

        // Architecture
        'arch.title': 'System Architecture',
        'arch.description': 'High-level design of the ModelEngine deployment on Ascend hardware.',
        'arch.layeredView': 'Layered View',
        'arch.pipeline': 'Data & Inference Pipeline',
        'arch.wsiInput': 'WSI Input',
        'arch.openslide': 'OpenSlide Read',
        'arch.preprocess': 'Preprocess',
        'arch.otsuMask': 'Otsu Mask & Cut',
        'arch.npuAccel': 'NPU Acceleration',
        'arch.modelEngine': 'ModelEngine',
        'arch.aippUni': 'AIPP + UNI (INT8)',
        'arch.aggregation': 'Aggregation',
        'arch.milDiagnosis': 'MIL & Diagnosis',

        // Stage titles
        'stage.connect': 'Connect 910B NPU & Device Context',
        'stage.connect.desc': 'Set device, build context, and check ModelEngine quota.',
        'stage.aipp': 'AIPP Preprocess Downlink',
        'stage.aipp.desc': 'Color normalization and mean/std offloaded to hardware AIPP.',
        'stage.atc': 'ATC Whole-Graph Compilation',
        'stage.atc.desc': 'Static shape + operator fusion to reduce DDR trips.',
        'stage.amct': 'AMCT Sparse-aware INT8 Calibration',
        'stage.amct.desc': 'Keep Attention FP16, quantize FFN to INT8 with KL search.',
        'stage.engine': 'ModelEngine Deployment & Routing',
        'stage.engine.desc': 'Register .om, expose REST/gRPC, enable multi-batch streaming.',
        'stage.monitor': 'Runtime Monitoring & Profiling',
        'stage.monitor.desc': 'Collect throughput, latency, and heat map traces in real time.',

        // Architecture Layers
        'layer.app.title': 'Application Layer',
        'layer.app.desc': 'End-user interfaces for pathologists and researchers',
        'layer.service.title': 'ModelEngine Service Layer',
        'layer.service.desc': 'Cloud orchestration, inference scheduling, and monitoring',
        'layer.infra.title': 'Infrastructure & Hardware Layer',
        'layer.infra.desc': 'Physical compute resources and storage',

        // Code Viewer
        'code.title': 'Project Explorer',
        'code.selectFile': 'Select a file to view source',
        'code.exploreDesc': 'Explore the implementation of the Data Preprocessing, UNI Backbone, and Ascend NPU Inference logic.',
        'code.noPreview': 'No preview available for this file.',
        'code.checkFile': 'Check other files for implementation details.',

        // Benchmark
        'benchmark.title': 'Benchmark Results',
        'benchmark.description': 'Performance evaluation comparing CPU baseline with Ascend NPU optimized versions.',
        'benchmark.tableTitle': 'Self-Evaluation Results Table',
        'benchmark.modelVersion': 'Model Version',
        'benchmark.patchAcc': 'Patch Top-1 Acc',
        'benchmark.wsiF1': 'WSI F1',
        'benchmark.throughput': 'Throughput (patch/s)',
        'benchmark.wsiLatency': 'Avg WSI Latency (ms)',
        'benchmark.notes': 'Notes',
        'benchmark.pending': 'Pending Test',
        'benchmark.baseline': 'CPU / PyTorch Baseline',
        'benchmark.baselineNote': 'No acceleration, control group',
        'benchmark.fp16': 'Ascend FP16 (No Quantization)',
        'benchmark.fp16Note': 'ATC compile + AIPP preprocessing',
        'benchmark.int8': 'Ascend INT8 (This Solution)',
        'benchmark.int8Note': 'AMCT mixed precision + Multi-Batch',

        // Dataset classes
        'class.normal': 'Normal',
        'class.benign': 'Benign',
        'class.insitu': 'In situ Carcinoma',
        'class.invasive': 'Invasive Carcinoma',
        'class.normalDesc': 'Normal glandular structure, no obvious lesions',
        'class.benignDesc': 'Benign lesions such as fibroadenoma',
        'class.insituDesc': 'In situ carcinoma confined to ducts',
        'class.invasiveDesc': 'Invasive carcinoma breaking through basement membrane',
    },
    zh: {
        // Sidebar
        'sidebar.title': 'ModelEngine',
        'sidebar.subtitle': 'UNI ViT 推理加速',
        'sidebar.overview': '概览与指标',
        'sidebar.architecture': '系统架构',
        'sidebar.code': '项目代码库',
        'sidebar.benchmark': '性能评测结果',
        'sidebar.deviceStatus': '设备状态',
        'sidebar.npuLoad': 'NPU 负载',

        // Dashboard
        'dashboard.title': '项目概览',
        'dashboard.description': '基于昇腾 NPU 加速 UNI 病理基础模型 (ViT) 推理。聚焦于使用 CANN、ATC 和 AMCT 技术对乳腺癌 WSI 切片进行分类（BACH 数据集）。',
        'dashboard.throughput': '吞吐量 (Ascend INT8)',
        'dashboard.throughputSub': '相比 CPU 基线提升 2700%',
        'dashboard.latency': '平均 WSI 时延',
        'dashboard.latencySub': '从约 15 分钟降低',
        'dashboard.accuracy': '模型精度 (F1)',
        'dashboard.accuracySub': '相比 FP32 损失 -0.4%',
        'dashboard.dataset': '数据集 (BACH)',
        'dashboard.datasetSub': '正常、良性、原位癌、浸润癌',
        'dashboard.throughputChart': '吞吐量对比 (Patch/秒)',
        'dashboard.latencyChart': '推理时延 (毫秒/批次)',
        'dashboard.datasetTitle': 'BACH 数据集分布',
        'dashboard.patchCount': 'Patch 数量',
        'dashboard.description2': '说明',
        'dashboard.controlPanel': '一键式控制面板',
        'dashboard.npuStatus': '昇腾 910B 可用性',
        'dashboard.checkButton': '检查可用性',
        'dashboard.monitoring': '实时监控',
        'dashboard.currentEngine': '当前加速引擎',
        'dashboard.engineInsights': '加速引擎与取舍提示',
        'dashboard.pipeline': '全流程跟踪',
        'dashboard.tradeoff': '时间自由度损耗',
        'dashboard.speedup': '相对基线加速比',
        'dashboard.liveLoad': '实时负载',
        'dashboard.temperature': '温度',
        'dashboard.stagesLabel': '流程阶段',
        'dashboard.ready': '就绪',
        'dashboard.running': '运行中',
        'dashboard.pending': '待处理',
        'dashboard.blocked': '阻塞',
        'dashboard.done': '完成',

        // Architecture
        'arch.title': '系统架构',
        'arch.description': 'ModelEngine 在昇腾硬件上部署的高层设计。',
        'arch.layeredView': '分层视图',
        'arch.pipeline': '数据与推理流水线',
        'arch.wsiInput': 'WSI 输入',
        'arch.openslide': 'OpenSlide 读取',
        'arch.preprocess': '预处理',
        'arch.otsuMask': 'Otsu 掩码与切分',
        'arch.npuAccel': 'NPU 加速',
        'arch.modelEngine': 'ModelEngine',
        'arch.aippUni': 'AIPP + UNI (INT8)',
        'arch.aggregation': '聚合',
        'arch.milDiagnosis': 'MIL 与诊断',

        // Stage titles
        'stage.connect': '连接 910B NPU 与设备上下文',
        'stage.connect.desc': '设置设备、构建上下文，并检查 ModelEngine 配额。',
        'stage.aipp': 'AIPP 预处理下沉',
        'stage.aipp.desc': '色彩归一化与均值方差在硬件 AIPP 侧完成。',
        'stage.atc': 'ATC 整图编译',
        'stage.atc.desc': '静态 Shape + 算子融合，减少 DDR 访问。',
        'stage.amct': 'AMCT 稀疏感知 INT8 校准',
        'stage.amct.desc': 'Attention 保持 FP16，FFN INT8，使用 KL 搜索阈值。',
        'stage.engine': 'ModelEngine 部署与路由',
        'stage.engine.desc': '注册 .om，开放 REST/gRPC，并启用多批次串流。',
        'stage.monitor': '运行监控与 Profiling',
        'stage.monitor.desc': '实时采集吞吐、时延与热点算子热力图。',

        // Architecture Layers
        'layer.app.title': '应用层',
        'layer.app.desc': '面向病理医生和研究人员的终端用户界面',
        'layer.service.title': 'ModelEngine 服务层',
        'layer.service.desc': '云端编排、推理调度与监控',
        'layer.infra.title': '基础设施与硬件层',
        'layer.infra.desc': '物理计算资源与存储',

        // Code Viewer
        'code.title': '项目浏览器',
        'code.selectFile': '选择文件查看源码',
        'code.exploreDesc': '浏览数据预处理、UNI 骨干网络和昇腾 NPU 推理逻辑的实现。',
        'code.noPreview': '此文件暂无预览。',
        'code.checkFile': '请查看其他文件了解实现细节。',

        // Benchmark
        'benchmark.title': '性能评测结果',
        'benchmark.description': '对比 CPU 基线与昇腾 NPU 优化版本的性能评估。',
        'benchmark.tableTitle': '自测结果记录表',
        'benchmark.modelVersion': '模型版本',
        'benchmark.patchAcc': 'Patch Top-1 准确率',
        'benchmark.wsiF1': 'WSI F1 分数',
        'benchmark.throughput': '吞吐量 (patch/s)',
        'benchmark.wsiLatency': '平均 WSI 时延 (ms)',
        'benchmark.notes': '备注',
        'benchmark.pending': '待实测',
        'benchmark.baseline': 'CPU / PyTorch 基线',
        'benchmark.baselineNote': '无加速，仅作对照',
        'benchmark.fp16': 'Ascend FP16（未量化）',
        'benchmark.fp16Note': 'ATC 编译 + AIPP 预处理下沉',
        'benchmark.int8': 'Ascend INT8（本方案）',
        'benchmark.int8Note': 'AMCT 混合精度量化 + Multi-Batch',

        // Dataset classes
        'class.normal': '正常',
        'class.benign': '良性',
        'class.insitu': '原位癌',
        'class.invasive': '浸润性癌',
        'class.normalDesc': '正常腺体结构，无明显病变',
        'class.benignDesc': '良性病变，如纤维腺瘤等',
        'class.insituDesc': '局限于导管内的原位癌',
        'class.invasiveDesc': '突破基底膜的浸润性癌',
    }
};

const I18nContext = createContext<I18nContextType | undefined>(undefined);

export const I18nProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const [language, setLanguage] = useState<Language>('zh');

    const t = (key: string): string => {
        return translations[language][key] || key;
    };

    return (
        <I18nContext.Provider value={{ language, setLanguage, t }}>
            {children}
        </I18nContext.Provider>
    );
};

export const useI18n = (): I18nContextType => {
    const context = useContext(I18nContext);
    if (!context) {
        throw new Error('useI18n must be used within an I18nProvider');
    }
    return context;
};

export default I18nProvider;
