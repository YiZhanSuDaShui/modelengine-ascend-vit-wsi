export interface FileNode {
  name: string;
  type: 'file' | 'folder';
  children?: FileNode[];
  content?: string; // Content for files
  language?: string;
}

export interface MetricData {
  name: string;
  throughput: number;
  latency: number;
  accuracy: number;
}

export type StageStatus = 'ready' | 'running' | 'pending' | 'blocked' | 'done';

export interface WorkflowStage {
  key: string;
  titleKey: string;
  descKey: string;
  status: StageStatus;
  eta?: string;
}

export interface EngineInsight {
  engine: string;
  mode: string;
  speedup: string;
  timeTradeoff: string;
  description: string;
}

export interface ArchitectureLayer {
  title: string;
  description: string;
  items: string[];
  color: string;
}