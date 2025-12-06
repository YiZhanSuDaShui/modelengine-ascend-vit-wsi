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

export interface ArchitectureLayer {
  title: string;
  description: string;
  items: string[];
  color: string;
}