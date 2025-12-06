import React from 'react';
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Architecture from './pages/Architecture';
import CodeViewer from './pages/CodeViewer';

const App: React.FC = () => {
  return (
    <HashRouter>
      <div className="flex h-screen bg-slate-50 text-slate-800 font-sans overflow-hidden">
        <Sidebar />
        <main className="flex-1 overflow-y-auto">
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/architecture" element={<Architecture />} />
            <Route path="/code" element={<CodeViewer />} />
          </Routes>
        </main>
      </div>
    </HashRouter>
  );
};

export default App;