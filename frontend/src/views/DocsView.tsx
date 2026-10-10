import React from 'react';
import { BookOpen, Terminal, CheckCircle2, Shield, Layers, FileText } from 'lucide-react';

export const DocsView: React.FC = () => {
  return (
    <div style={{ maxWidth: '1000px', margin: '0 auto', padding: '30px 20px' }}>
      <div style={{ marginBottom: '32px' }}>
        <h2 style={{ fontSize: '28px', fontWeight: '700', marginBottom: '8px' }}>Platform Documentation</h2>
        <p style={{ color: '#94a3b8', fontSize: '14px' }}>
          Architectural principles, mathematical formulations, and reproduction procedures for Stock-AI.
        </p>
      </div>

      {/* Section 1: Target Definition */}
      <div className="glass-panel" style={{ padding: '28px', marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '14px' }}>
          <Layers size={20} color="#38bdf8" />
          <h3 style={{ fontSize: '18px', fontWeight: '600' }}>1. Target Definition & Lookahead Prevention</h3>
        </div>
        <p style={{ fontSize: '14px', color: '#cbd5e1', lineHeight: '1.7', marginBottom: '12px' }}>
          The prediction target <span style={{ fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>y<sub>t</sub> &isin; &#123;0, 1&#125;</span> represents the binary price trend of trading day <span style={{ fontFamily: 'var(--font-mono)' }}>t</span> relative to day <span style={{ fontFamily: 'var(--font-mono)' }}>t-1</span>:
        </p>
        <div style={{ background: '#0a0f1d', padding: '12px 18px', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.08)', fontFamily: 'var(--font-mono)', fontSize: '13px', color: '#38bdf8', marginBottom: '14px' }}>
          Diff<sub>t</sub> = 1 &nbsp;if&nbsp; Close<sub>t</sub> &gt; Close<sub>t-1</sub> &nbsp;else&nbsp; 0
        </div>
        <p style={{ fontSize: '14px', color: '#94a3b8', lineHeight: '1.7' }}>
          Under our strict leakage-free protocol, the market sequence input strictly terminates on trading day <span style={{ fontFamily: 'var(--font-mono)', color: '#a5b4fc' }}>t-1</span>. Under no circumstances do current-day open, high, low, or close prices enter the feature array prior to forecasting <span style={{ fontFamily: 'var(--font-mono)' }}>y<sub>t</sub></span>.
        </p>
      </div>

      {/* Section 2: Chronological Splitting */}
      <div className="glass-panel" style={{ padding: '28px', marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '14px' }}>
          <Shield size={20} color="#10b981" />
          <h3 style={{ fontSize: '18px', fontWeight: '600' }}>2. Chronological Partitioning Protocol</h3>
        </div>
        <p style={{ fontSize: '14px', color: '#cbd5e1', lineHeight: '1.7', marginBottom: '12px' }}>
          Financial time series exhibit autocorrelation and non-stationarity. Random cross-validation leaks future regime information into the past. Stock-AI enforces three strict chronological splits:
        </p>
        <ul style={{ listStyle: 'none', padding: 0, fontSize: '14px', color: '#94a3b8', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <li style={{ display: 'flex', gap: '8px' }}>
            <span style={{ color: '#38bdf8', fontWeight: '600' }}>• Training Horizon:</span> 2010-02-22 to 2022-12-31 (~3,200 sessions).
          </li>
          <li style={{ display: 'flex', gap: '8px' }}>
            <span style={{ color: '#818cf8', fontWeight: '600' }}>• Validation Horizon:</span> 2023-01-01 to 2024-06-30 (~375 sessions). Used solely for early stopping and threshold calibration.
          </li>
          <li style={{ display: 'flex', gap: '8px' }}>
            <span style={{ color: '#34d399', fontWeight: '600' }}>• Held-Out Test Horizon:</span> 2024-07-01 to 2026-10-09 (572 sessions). Evaluated strictly once.
          </li>
        </ul>
      </div>

      {/* Section 3: Commands */}
      <div className="glass-panel" style={{ padding: '28px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '14px' }}>
          <Terminal size={20} color="#f59e0b" />
          <h3 style={{ fontSize: '18px', fontWeight: '600' }}>3. Reproduction Commands</h3>
        </div>
        <div style={{ background: '#0a0f1d', padding: '16px', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.08)', fontFamily: 'var(--font-mono)', fontSize: '12px', color: '#f8fafc', display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div><span style={{ color: '#64748b' }}># Run full test suite</span><br />pytest tests/</div>
          <div><span style={{ color: '#64748b' }}># Refresh market data for all 12 tickers</span><br />python scripts/refresh_data.py</div>
          <div><span style={{ color: '#64748b' }}># Validate dataset schema, boundaries, and zero NaNs</span><br />python scripts/validate_data.py</div>
          <div><span style={{ color: '#64748b' }}># Run benchmarks across baselines and BiLSTM</span><br />python scripts/evaluate.py</div>
          <div><span style={{ color: '#64748b' }}># Launch FastAPI backend service</span><br />uvicorn backend.main:app --port 8000 --reload</div>
        </div>
      </div>
    </div>
  );
};
