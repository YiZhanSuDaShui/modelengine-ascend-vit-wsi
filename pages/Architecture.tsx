import React from 'react';
import { Layers, ArrowRight, Database, Cpu, Server } from 'lucide-react';
import { LAYERS } from '../constants';

const Architecture: React.FC = () => {
  return (
    <div className="p-8 space-y-12">
      <div>
        <h2 className="text-3xl font-bold text-slate-900">System Architecture</h2>
        <p className="text-slate-500 mt-2">
          High-level design of the ModelEngine deployment on Ascend hardware.
        </p>
      </div>

      {/* Layered Architecture Diagram */}
      <div className="space-y-4">
        <h3 className="text-xl font-semibold text-slate-800 flex items-center gap-2">
          <Layers className="text-blue-600" /> Layered View
        </h3>
        <div className="space-y-4 max-w-4xl mx-auto">
          {LAYERS.map((layer, index) => (
            <div key={index} className={`p-6 rounded-xl border-2 ${layer.color} shadow-sm transition-transform hover:scale-[1.01]`}>
              <div className="flex justify-between items-start mb-4">
                <h4 className="font-bold text-lg">{layer.title}</h4>
                <span className="text-xs font-mono uppercase bg-white/50 px-2 py-1 rounded">Layer 0{index + 1}</span>
              </div>
              <p className="text-sm mb-4 opacity-80">{layer.description}</p>
              <div className="flex gap-3 flex-wrap">
                {layer.items.map((item, i) => (
                  <div key={i} className="bg-white/80 px-3 py-1.5 rounded text-sm font-medium shadow-sm">
                    {item}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Process Flow Diagram */}
      <div className="space-y-6">
        <h3 className="text-xl font-semibold text-slate-800 flex items-center gap-2">
          <Database className="text-blue-600" /> Data & Inference Pipeline
        </h3>
        
        <div className="bg-white p-8 rounded-xl border border-slate-200 shadow-sm overflow-x-auto">
          <div className="flex items-center min-w-[800px] justify-between text-center">
            
            {/* Step 1 */}
            <div className="flex flex-col items-center gap-3 w-40">
              <div className="w-16 h-16 bg-slate-100 rounded-lg flex items-center justify-center border border-slate-300">
                <Database className="text-slate-500" />
              </div>
              <div>
                <h5 className="font-bold text-slate-800">WSI Input</h5>
                <p className="text-xs text-slate-500">OpenSlide Read</p>
              </div>
            </div>

            <ArrowRight className="text-slate-300" />

            {/* Step 2 */}
            <div className="flex flex-col items-center gap-3 w-40">
              <div className="w-16 h-16 bg-indigo-50 rounded-lg flex items-center justify-center border border-indigo-200">
                <span className="text-xs font-bold text-indigo-600">Patching</span>
              </div>
              <div>
                <h5 className="font-bold text-slate-800">Preprocess</h5>
                <p className="text-xs text-slate-500">Otsu Mask & Cut</p>
              </div>
            </div>

            <ArrowRight className="text-slate-300" />

            {/* Step 3 */}
            <div className="flex flex-col items-center gap-3 w-40">
              <div className="w-16 h-16 bg-red-50 rounded-lg flex items-center justify-center border border-red-200 shadow-md">
                <Cpu className="text-red-600" />
              </div>
              <div className="relative">
                <span className="absolute -top-12 left-1/2 -translate-x-1/2 bg-red-600 text-white text-[10px] px-2 py-0.5 rounded-full">
                  NPU Acceleration
                </span>
                <h5 className="font-bold text-slate-800">ModelEngine</h5>
                <p className="text-xs text-slate-500">AIPP + UNI (INT8)</p>
              </div>
            </div>

            <ArrowRight className="text-slate-300" />

            {/* Step 4 */}
            <div className="flex flex-col items-center gap-3 w-40">
              <div className="w-16 h-16 bg-green-50 rounded-lg flex items-center justify-center border border-green-200">
                <Server className="text-green-600" />
              </div>
              <div>
                <h5 className="font-bold text-slate-800">Aggregation</h5>
                <p className="text-xs text-slate-500">MIL & Diagnosis</p>
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
};

export default Architecture;