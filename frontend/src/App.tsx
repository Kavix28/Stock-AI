import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { LandingView } from './views/LandingView';
import { DashboardView } from './views/DashboardView';
import { ExplorerView } from './views/ExplorerView';
import { ModelLabView } from './views/ModelLabView';
import { PredictionDemoView } from './views/PredictionDemoView';
import { DocsView } from './views/DocsView';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState('landing');

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar currentTab={currentTab} setCurrentTab={setCurrentTab} />

      <main style={{ flex: 1 }}>
        {currentTab === 'landing' && <LandingView onNavigate={setCurrentTab} />}
        {currentTab === 'dashboard' && <DashboardView />}
        {currentTab === 'explorer' && <ExplorerView />}
        {currentTab === 'lab' && <ModelLabView />}
        {currentTab === 'demo' && <PredictionDemoView />}
        {currentTab === 'docs' && <DocsView />}
      </main>

      <footer style={{
        borderTop: '1px solid rgba(255, 255, 255, 0.05)',
        background: '#070b14',
        padding: '24px 20px',
        textAlign: 'center',
        fontSize: '12px',
        color: '#64748b'
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            Stock-AI &copy; 2026 &bull; Multimodal AI for Stock Trend Prediction
          </div>
          <div style={{ display: 'flex', gap: '16px' }}>
            <span>DOI: 10.1007/s00521-024-10303-1</span>
            <span>&bull;</span>
            <span style={{ color: '#94a3b8' }}>Academic Research &mdash; Not Financial Advice</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
