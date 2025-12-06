import React, { useState } from 'react';
import { Folder, FileCode, ChevronRight, ChevronDown, FileJson, Terminal, Code } from 'lucide-react';
import { PROJECT_STRUCTURE } from '../constants';
import { FileNode } from '../types';

const getIcon = (name: string) => {
  if (name.endsWith('.py')) return <FileCode size={16} className="text-blue-500" />;
  if (name.endsWith('.json') || name.endsWith('.yaml')) return <FileJson size={16} className="text-yellow-500" />;
  if (name.endsWith('.sh')) return <Terminal size={16} className="text-green-500" />;
  return <FileCode size={16} className="text-slate-400" />;
};

const TreeNode: React.FC<{ 
  node: FileNode; 
  level: number; 
  onSelect: (node: FileNode) => void;
  selectedPath: string;
  currentPath: string;
}> = ({ node, level, onSelect, selectedPath, currentPath }) => {
  const [isOpen, setIsOpen] = useState(level < 1);
  const fullPath = `${currentPath}/${node.name}`;
  const isSelected = fullPath === selectedPath;

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (node.type === 'folder') {
      setIsOpen(!isOpen);
    } else {
      onSelect(node);
      // Trigger update of selected path logic in parent (simplified here via prop drilling usually, but state is top level)
    }
  };

  return (
    <div>
      <div 
        onClick={handleClick}
        className={`flex items-center gap-2 py-1.5 px-2 cursor-pointer hover:bg-slate-100 rounded text-sm transition-colors ${isSelected ? 'bg-blue-100 text-blue-700 font-medium' : 'text-slate-600'}`}
        style={{ paddingLeft: `${level * 16 + 8}px` }}
      >
        {node.type === 'folder' && (
          <span className="text-slate-400">
            {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </span>
        )}
        {node.type === 'folder' ? <Folder size={16} className="text-slate-400" /> : getIcon(node.name)}
        <span>{node.name}</span>
      </div>
      {isOpen && node.children && (
        <div>
          {node.children.map((child) => (
            <TreeNode 
              key={child.name} 
              node={child} 
              level={level + 1} 
              onSelect={onSelect}
              selectedPath={selectedPath}
              currentPath={fullPath}
            />
          ))}
        </div>
      )}
    </div>
  );
};

const CodeViewer: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<FileNode | null>(null);

  // Helper to find initial state or track selection
  const handleSelect = (node: FileNode) => {
    setSelectedFile(node);
  };

  return (
    <div className="flex h-full">
      {/* File Tree */}
      <div className="w-80 border-r border-slate-200 bg-white flex flex-col h-full overflow-hidden">
        <div className="p-4 border-b border-slate-100 bg-slate-50">
          <h3 className="font-bold text-sm text-slate-700">Project Explorer</h3>
        </div>
        <div className="flex-1 overflow-y-auto py-2">
          <TreeNode 
            node={PROJECT_STRUCTURE} 
            level={0} 
            onSelect={handleSelect}
            selectedPath={selectedFile ? `/${selectedFile.name}` : ''} // Simplified path matching
            currentPath=""
          />
        </div>
      </div>

      {/* Code Content */}
      <div className="flex-1 bg-slate-50 h-full overflow-hidden flex flex-col">
        {selectedFile ? (
          <>
             <div className="bg-white border-b border-slate-200 px-6 py-3 flex items-center justify-between shadow-sm">
                <div className="flex items-center gap-2">
                   {getIcon(selectedFile.name)}
                   <span className="font-mono text-sm font-semibold text-slate-700">{selectedFile.name}</span>
                </div>
                <div className="text-xs text-slate-400 uppercase font-bold">{selectedFile.language || 'text'}</div>
             </div>
             <div className="flex-1 overflow-auto p-6">
               {selectedFile.content ? (
                 <pre className="font-mono text-sm leading-relaxed text-slate-800 bg-white p-6 rounded-lg border border-slate-200 shadow-sm">
                   <code>{selectedFile.content}</code>
                 </pre>
               ) : (
                 <div className="flex flex-col items-center justify-center h-full text-slate-400">
                   <FileCode size={48} className="mb-4 opacity-50" />
                   <p>No preview available for this file.</p>
                   <p className="text-xs mt-2">Check 'ascend_infer.py' or 'atc_convert.sh' for details.</p>
                 </div>
               )}
             </div>
          </>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-slate-400">
            <Code size={64} className="mb-6 opacity-20" />
            <h3 className="text-lg font-medium text-slate-500">Select a file to view source</h3>
            <p className="max-w-md text-center mt-2 text-slate-400">
              Explore the implementation of the Data Preprocessing, UNI Backbone, and Ascend NPU Inference logic.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default CodeViewer;