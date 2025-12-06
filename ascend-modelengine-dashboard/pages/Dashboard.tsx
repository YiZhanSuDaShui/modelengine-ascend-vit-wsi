import React from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer,
  LineChart, Line 
} from 'recharts';
import { Activity, Zap, Clock, FileText } from 'lucide-react';
import { PERFORMANCE_DATA } from '../constants';

const MetricCard: React.FC<{ title: string; value: string; sub: string; icon: React.ReactNode; color: string }> = ({ title, value, sub, icon, color }) => (
  <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
    <div className="flex items-start justify-between">
      <div>
        <p className="text-sm font-medium text-slate-500 mb-1">{title}</p>
        <h3 className="text-2xl font-bold text-slate-800">{value}</h3>
        <p className={`text-xs mt-1 font-medium ${color}`}>{sub}</p>
      </div>
      <div className={`p-3 rounded-lg bg-slate-50 text-slate-600`}>
        {icon}
      </div>
    </div>
  </div>
);

const Dashboard: React.FC = () => {
  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-bold text-slate-900">Project Overview</h2>
        <p className="text-slate-500 mt-2 max-w-3xl">
          Accelerating UNI Pathological Foundation Model (ViT) Inference on Ascend NPU. 
          Focusing on WSI Patch classification for Breast Cancer (BACH dataset) using CANN, ATC, and AMCT technologies.
        </p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <MetricCard 
          title="Throughput (Ascend INT8)" 
          value="420 patch/s" 
          sub="+1400% vs CPU Baseline" 
          color="text-green-600"
          icon={<Zap size={24} />}
        />
        <MetricCard 
          title="Avg WSI Latency" 
          value="1.2 mins" 
          sub="Reduced from ~15 mins" 
          color="text-blue-600"
          icon={<Clock size={24} />}
        />
        <MetricCard 
          title="Model Accuracy (F1)" 
          value="0.881" 
          sub="-0.4% Loss (vs FP32)" 
          color="text-orange-600"
          icon={<Activity size={24} />}
        />
        <MetricCard 
          title="Dataset (BACH)" 
          value="4 Classes" 
          sub="Normal, Benign, InSitu, Invasive" 
          color="text-purple-600"
          icon={<FileText size={24} />}
        />
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Throughput Chart */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <h3 className="text-lg font-bold text-slate-800 mb-6">Throughput Comparison (Patch/s)</h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={PERFORMANCE_DATA} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 12}} />
                <YAxis axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 12}} />
                <RechartsTooltip 
                  cursor={{fill: '#f1f5f9'}}
                  contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}}
                />
                <Bar dataKey="throughput" fill="#dc2626" radius={[4, 4, 0, 0]} name="Throughput (img/s)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Latency Chart */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <h3 className="text-lg font-bold text-slate-800 mb-6">Inference Latency (ms/batch)</h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={PERFORMANCE_DATA} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 12}} />
                <YAxis axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 12}} />
                <RechartsTooltip 
                   contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}}
                />
                <Line type="monotone" dataKey="latency" stroke="#2563eb" strokeWidth={3} dot={{r: 6}} name="Latency (ms)" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;