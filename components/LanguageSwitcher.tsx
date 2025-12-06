import React from 'react';
import { useI18n, Language } from '../i18n';
import { Globe } from 'lucide-react';

const LanguageSwitcher: React.FC = () => {
    const { language, setLanguage } = useI18n();

    const toggleLanguage = () => {
        setLanguage(language === 'en' ? 'zh' : 'en');
    };

    return (
        <button
            onClick={toggleLanguage}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition-colors text-sm font-medium"
            title={language === 'en' ? '切换到中文' : 'Switch to English'}
        >
            <Globe size={16} className="text-slate-400" />
            <span className="text-white">{language === 'en' ? 'EN' : '中文'}</span>
            <span className="text-slate-500">|</span>
            <span className="text-slate-400">{language === 'en' ? '中文' : 'EN'}</span>
        </button>
    );
};

export default LanguageSwitcher;
