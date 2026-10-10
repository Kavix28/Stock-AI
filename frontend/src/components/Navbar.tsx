import React from 'react';
import { Activity, BarChart2, TrendingUp, Cpu, Compass, BookOpen, ShieldAlert } from 'lucide-react';

interface NavbarProps {
  currentTab: string;
  setCurrentTab: (tab: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({ currentTab, setCurrentTab }) => {
  const tabs = [
    { id: 'landing', label: 'Overview', icon: Compass },
    { id: 'dashboard', label: 'Research Dashboard', icon: BarChart2 },
    { id: 'explorer', label: 'Stock Explorer', icon: TrendingUp },
    { id: 'lab', label: 'Model Laboratory', icon: Cpu },
    { id: 'demo', label: 'Prediction Demo', icon: Activity },
    { id: 'docs', label: 'Documentation', icon: BookOpen },
  ];

  return (
    <header style={{
      borderBottom: '1px solid rgba(99, 102, 241, 0.2)',
      background: 'rgba(7, 11, 20, 0.85)',
      backdropFilter: 'blur(16px)',
      position: 'sticky',
      top: 0,
      zIndex: 50,
      padding: '0 24px'
    }}>
      <div style={{
        maxWidth: '1400px',
        margin: '0 auto',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        height: '68px'
      }}>
        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer' }} onClick={() => setCurrentTab('landing')}>
          <div style={{
            width: '38px',
            height: '38px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #6366f1, #38bdf8)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 16px rgba(99, 102, 241, 0.4)'
          }}>
            <Activity size={22} color="#ffffff" />
          </div>
          <div>
            <div style={{ fontSize: '18px', fontWeight: '700', letterSpacing: '-0.3px', color: '#fff' }}>
              Stock<span style={{ color: '#38bdf8' }}>-AI</span>
            </div>
            <div style={{ fontSize: '10px', color: '#94a3b8', letterSpacing: '0.5px', textTransform: 'uppercase' }}>
              Multimodal Research Platform
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav style={{ display: 'flex', gap: '6px' }}>
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const active = currentTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setCurrentTab(tab.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '8px 14px',
                  borderRadius: '8px',
                  border: 'none',
                  fontSize: '13px',
                  fontWeight: active ? '600' : '500',
                  color: active ? '#ffffff' : '#94a3b8',
                  background: active ? 'rgba(99, 102, 241, 0.2)' : 'transparent',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                  borderBottom: active ? '2px solid #6366f1' : '2px solid transparent'
                }}
              >
                <Icon size={16} color={active ? '#38bdf8' : '#64748b'} />
                {tab.label}
              </button>
            );
          })}
        </nav>

        {/* Paper DOI Badge */}
        <a
          href="https://doi.org/10.1007/s00521-024-10303-1"
          target="_blank"
          rel="noopener noreferrer"
          className="badge badge-indigo"
          style={{ textDecoration: 'none' }}
        >
          DOI: 10.1007/s00521-024-10303-1
        </a>
      </div>
    </header>
  );
};
