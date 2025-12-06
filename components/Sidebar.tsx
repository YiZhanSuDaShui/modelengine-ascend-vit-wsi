import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Network, Code, Cpu, Globe2 } from 'lucide-react';
import { useI18n } from '../i18n';

const Sidebar: React.FC = () => {
  const { t, language, setLanguage } = useI18n();

  const navItems = [
    { path: '/dashboard', label: t('sidebar.overview'), icon: <LayoutDashboard size={20} /> },
    { path: '/architecture', label: t('sidebar.architecture'), icon: <Network size={20} /> },
    { path: '/code', label: t('sidebar.code'), icon: <Code size={20} /> },
  ];

  return (
    <aside className="w-64 bg-slate-900 text-white flex flex-col h-full shadow-xl">
      <div className="p-6 border-b border-slate-800 flex items-center space-x-3">
        <div className="bg-red-600 p-2 rounded-lg">
           <Cpu size={24} className="text-white" />
        </div>
        <div>
          <h1 className="text-lg font-bold leading-tight">{t('sidebar.title')}</h1>
          <p className="text-xs text-slate-400">{t('sidebar.subtitle')}</p>
        </div>
      </div>

      <nav className="flex-1 p-4 space-y-2">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              `flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                isActive
                  ? 'bg-red-600 text-white shadow-md'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-white'
              }`
            }
          >
            {item.icon}
            <span className="font-medium">{item.label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="p-6 border-t border-slate-800 space-y-4">
        <div className="bg-slate-800 rounded-lg p-4 flex items-center justify-between text-sm">
          <div className="flex items-center gap-2 text-slate-300">
            <Globe2 size={16} />
            <span>Bilingual</span>
          </div>
          <div className="flex gap-2">
            <button
              className={`px-3 py-1 rounded text-xs font-semibold border ${language === 'zh' ? 'bg-red-600 border-red-500 text-white' : 'bg-slate-900 border-slate-700 text-slate-300'}`}
              onClick={() => setLanguage('zh')}
            >
              中文
            </button>
            <button
              className={`px-3 py-1 rounded text-xs font-semibold border ${language === 'en' ? 'bg-red-600 border-red-500 text-white' : 'bg-slate-900 border-slate-700 text-slate-300'}`}
              onClick={() => setLanguage('en')}
            >
              EN
            </button>
          </div>
        </div>

        <div className="bg-slate-800 rounded-lg p-4">
          <p className="text-xs text-slate-400 uppercase font-semibold mb-2">{t('sidebar.deviceStatus')}</p>
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm font-medium">Ascend 910B</span>
            <span className="flex h-2 w-2 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
            </span>
          </div>
          <div className="w-full bg-slate-700 rounded-full h-1.5 mt-2">
            <div className="bg-green-500 h-1.5 rounded-full" style={{ width: '35%' }}></div>
          </div>
          <p className="text-[10px] text-slate-500 mt-1">{t('sidebar.npuLoad')}: 35%</p>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;