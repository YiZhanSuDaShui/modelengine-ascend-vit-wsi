import React, { useEffect, useMemo, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  LineChart,
  Line
} from 'recharts';
import { Activity, Zap, Clock, FileText, Wifi, Gauge, Workflow, Sparkles, CheckCircle2, AlertTriangle } from 'lucide-react';
import { PERFORMANCE_DATA, WORKFLOW_STAGES, ENGINE_INSIGHTS } from '../constants';
import { WorkflowStage } from '../types';
import { useI18n } from '../i18n';

const statusColor: Record<string, string> = {
  ready: 'bg-green-100 text-green-700',
  running: 'bg-blue-100 text-blue-700',
  pending: 'bg-amber-100 text-amber-700',
  blocked: 'bg-red-100 text-red-700',
  done: 'bg-emerald-100 text-emerald-700'
};

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
  const { t } = useI18n();
  const [npuStatus, setNpuStatus] = useState<'ready' | 'checking' | 'error'>('ready');
  const [activeStage, setActiveStage] = useState(1);
  const [telemetry, setTelemetry] = useState({ load: 37, temperature: 46, throughput: 420, latency: 14 });
  const [engineIndex, setEngineIndex] = useState(1);

  const handleCheckNpu = () => {
    setNpuStatus('checking');
    setTimeout(() => {
      setNpuStatus('ready');
    }, 900);
  };

  useEffect(() => {
    const tick = setInterval(() => {
      setTelemetry((prev) => ({
        load: Math.min(95, Math.max(18, prev.load + (Math.random() * 10 - 5))),
        temperature: Math.min(75, Math.max(40, prev.temperature + (Math.random() * 4 - 2))),
        throughput: 410 + Math.round(Math.random() * 18),
        latency: Number((13.5 + Math.random()).toFixed(1))
      }));
      setEngineIndex((prev) => (prev + 1) % ENGINE_INSIGHTS.length);
      setActiveStage((prev) => (prev + 1) % WORKFLOW_STAGES.length);
    }, 3200);
    return () => clearInterval(tick);
  }, []);

  const stagedData: WorkflowStage[] = useMemo(() =>
    WORKFLOW_STAGES.map((stage, index) => {
      if (index < activeStage) return { ...stage, status: 'done' };
      if (index === activeStage) return { ...stage, status: 'running' };
      return { ...stage, status: 'pending' };
    }),
  [activeStage]);

  const currentEngine = ENGINE_INSIGHTS[engineIndex];

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-start justify-between gap-6">
        <div>
          <h2 className="text-3xl font-bold text-slate-900 flex items-center gap-3">
            <Sparkles className="text-red-600" /> {t('dashboard.title')}
          </h2>
          <p className="text-slate-500 mt-2 max-w-4xl">
            {t('dashboard.description')}
          </p>
        </div>
        <div className="bg-white border border-slate-200 px-5 py-4 rounded-xl shadow-sm text-sm text-slate-600 max-w-xs">
          <div className="flex items-center justify-between mb-2">
            <span className="font-semibold">Ascend 910B</span>
            <span className="flex h-2.5 w-2.5 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-green-500"></span>
            </span>
          </div>
          <p className="text-xs text-slate-500 leading-relaxed">
            {t('dashboard.controlPanel')} · ModelEngine · ATC/AMCT · AIPP
          </p>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <MetricCard
          title={t('dashboard.throughput')}
          value={`${telemetry.throughput} patch/s`}
          sub="+1400% vs CPU Baseline"
          color="text-green-600"
          icon={<Zap size={24} />}
        />
        <MetricCard
          title={t('dashboard.latency')}
          value={`${telemetry.latency} ms/batch`}
          sub={t('dashboard.latencySub')}
          color="text-blue-600"
          icon={<Clock size={24} />}
        />
        <MetricCard
          title={t('dashboard.accuracy')}
          value="0.881"
          sub={t('dashboard.accuracySub')}
          color="text-orange-600"
          icon={<Activity size={24} />}
        />
        <MetricCard
          title={t('dashboard.dataset')}
          value={t('dashboard.datasetSub')}
          sub="BACH · 4 classes"
          color="text-purple-600"
          icon={<FileText size={24} />}
        />
      </div>

      {/* Control + Telemetry */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-xs uppercase font-semibold text-slate-500">{t('dashboard.pipeline')}</p>
              <h3 className="text-xl font-bold text-slate-800">{t('dashboard.npuStatus')}</h3>
            </div>
            <button
              onClick={handleCheckNpu}
              className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg shadow-sm text-sm hover:bg-red-700 transition"
            >
              <Wifi size={16} /> {t('dashboard.checkButton')}
            </button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 rounded-lg border border-slate-200 bg-slate-50">
              <p className="text-xs uppercase text-slate-500 font-semibold mb-1">{t('dashboard.monitoring')}</p>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-500">910B</p>
                  <p className="text-lg font-bold text-slate-800">{npuStatus === 'checking' ? 'Checking…' : 'Online'}</p>
                  <p className="text-xs text-slate-500">{t('dashboard.liveLoad')}: {telemetry.load.toFixed(0)}%</p>
                </div>
                <div className={`px-3 py-2 rounded-lg text-xs font-semibold ${npuStatus === 'ready' ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'}`}>
                  {npuStatus === 'ready' ? t('dashboard.ready') : 'Syncing'}
                </div>
              </div>
            </div>
            <div className="p-4 rounded-lg border border-slate-200 bg-slate-50">
              <p className="text-xs uppercase text-slate-500 font-semibold mb-1">{t('dashboard.currentEngine')}</p>
              <p className="text-lg font-bold text-slate-800">{currentEngine.engine}</p>
              <p className="text-xs text-slate-500">{currentEngine.mode}</p>
            </div>
            <div className="p-4 rounded-lg border border-slate-200 bg-slate-50">
              <p className="text-xs uppercase text-slate-500 font-semibold mb-1">{t('dashboard.tradeoff')}</p>
              <p className="text-lg font-bold text-slate-800">{currentEngine.timeTradeoff}</p>
              <p className="text-xs text-slate-500">{t('dashboard.speedup')}: {currentEngine.speedup}</p>
            </div>
          </div>

          <div className="mt-6 space-y-3">
            <div className="flex items-center gap-2 text-slate-700 font-semibold text-sm">
              <Workflow size={16} className="text-red-600" /> {t('dashboard.stagesLabel')}
            </div>
            <div className="space-y-2">
              {stagedData.map((stage) => (
                <div key={stage.key} className="p-3 rounded-lg border border-slate-200 bg-white flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {stage.status === 'done' ? <CheckCircle2 className="text-green-600" size={18} /> : <AlertTriangle className="text-amber-500" size={18} />}
                    <div>
                      <p className="text-sm font-semibold text-slate-800">{t(stage.titleKey)}</p>
                      <p className="text-xs text-slate-500">{t(stage.descKey)}</p>
                    </div>
                  </div>
                  <div className={`text-[11px] px-3 py-1 rounded-full font-semibold ${statusColor[stage.status]}`}>
                    {t(`dashboard.${stage.status}`)} · {stage.eta}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-xs uppercase font-semibold text-slate-500">{t('dashboard.monitoring')}</p>
              <h3 className="text-xl font-bold text-slate-800">{t('dashboard.currentEngine')}</h3>
            </div>
            <Gauge className="text-red-600" />
          </div>
          <div className="space-y-3 text-sm text-slate-600">
            <div className="flex items-center justify-between">
              <span>{t('dashboard.liveLoad')}</span>
              <span className="font-semibold">{telemetry.load.toFixed(0)}%</span>
            </div>
            <div className="flex items-center justify-between">
              <span>{t('dashboard.temperature')}</span>
              <span className="font-semibold">{telemetry.temperature.toFixed(1)}°C</span>
            </div>
            <div className="flex items-center justify-between">
              <span>{t('dashboard.speedup')}</span>
              <span className="font-semibold">{currentEngine.speedup}</span>
            </div>
            <div className="flex items-center justify-between">
              <span>{t('dashboard.tradeoff')}</span>
              <span className="font-semibold">{currentEngine.timeTradeoff}</span>
            </div>
            <div className="text-xs text-slate-500 bg-slate-50 p-3 rounded-lg border border-slate-200 leading-relaxed">
              {currentEngine.description}
            </div>
          </div>
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <h3 className="text-lg font-bold text-slate-800 mb-6">{t('dashboard.throughputChart')}</h3>
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

        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <h3 className="text-lg font-bold text-slate-800 mb-6">{t('dashboard.latencyChart')}</h3>
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

      {/* Accelerator Insights */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
        <div className="flex items-center gap-2 mb-4">
          <Gauge className="text-red-600" />
          <h3 className="text-lg font-bold text-slate-800">{t('dashboard.engineInsights')}</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {ENGINE_INSIGHTS.map((insight) => (
            <div key={insight.engine} className="p-4 rounded-lg border border-slate-200 bg-slate-50">
              <p className="text-sm font-bold text-slate-800">{insight.engine}</p>
              <p className="text-xs text-slate-500 mb-2">{insight.mode}</p>
              <div className="flex items-center justify-between text-xs text-slate-600">
                <span>{t('dashboard.speedup')}</span>
                <span className="font-semibold text-green-600">{insight.speedup}</span>
              </div>
              <div className="flex items-center justify-between text-xs text-slate-600 mt-1">
                <span>{t('dashboard.tradeoff')}</span>
                <span className="font-semibold text-amber-600">{insight.timeTradeoff}</span>
              </div>
              <p className="text-xs text-slate-500 mt-3 leading-relaxed">{insight.description}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
