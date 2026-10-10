import React, { useState } from 'react';
import { requestPrediction, PredictionResult } from '../api';
import {
  Activity,
  ArrowUpRight,
  ArrowDownRight,
  ShieldAlert,
  Sparkles,
  CheckCircle,
  Cpu,
  Layers,
  Sliders,
  Calendar,
  FileText
} from 'lucide-react';

export const PredictionDemoView: React.FC = () => {
  const tickers = ['AAPL', 'AMT', 'BIO', 'JNJ', 'NDAQ', 'NFLX', 'NVDA', 'PLD', 'SBUX', 'SCHW', 'TSLA', 'ATVI'];
  const [selectedTicker, setSelectedTicker] = useState('AAPL');
  const [modelType, setModelType] = useState<'gmu_fusion' | 'market_bilstm'>('gmu_fusion');
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);

  const runPrediction = async () => {
    setLoading(true);
    const res = await requestPrediction(selectedTicker, modelType);
    setPrediction(res);
    setLoading(false);
  };

  const marketPct = prediction?.market_gate_weight !== undefined ? Math.round(prediction.market_gate_weight * 100) : 50;
  const newsPct = prediction?.news_gate_weight !== undefined ? Math.round(prediction.news_gate_weight * 100) : 50;

  return (
    <div style={{ maxWidth: '1050px', margin: '0 auto', padding: '30px 20px' }}>
      <div style={{ textAlign: 'center', marginBottom: '32px' }}>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', padding: '6px 14px', borderRadius: '20px', background: 'rgba(99, 102, 241, 0.1)', border: '1px solid rgba(99, 102, 241, 0.25)', color: '#a5b4fc', fontSize: '13px', fontWeight: 600, marginBottom: '12px' }}>
          <Sparkles size={14} /> Multimodal Neural Inference Engine
        </div>
        <h2 style={{ fontSize: '30px', fontWeight: '800', marginBottom: '10px', color: '#f8fafc' }}>
          Multimodal Research Inference & Gate Analysis
        </h2>
        <p style={{ color: '#94a3b8', fontSize: '14px', maxWidth: '680px', margin: '0 auto', lineHeight: '1.6' }}>
          Evaluate verified out-of-sample predictions combining 20-day market time-series encoders with FinBERT news representations via learned Gated Multimodal Units (GMU).
        </p>
      </div>

      {/* Control Panel */}
      <div className="glass-panel" style={{ padding: '24px', marginBottom: '28px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '20px', alignItems: 'flex-end' }}>
          {/* Ticker Selector */}
          <div>
            <label style={{ display: 'block', fontSize: '12px', color: '#94a3b8', marginBottom: '8px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Equity Ticker
            </label>
            <select
              value={selectedTicker}
              onChange={(e) => setSelectedTicker(e.target.value)}
              style={{
                width: '100%',
                padding: '12px 16px',
                borderRadius: '8px',
                background: '#0f172a',
                border: '1px solid rgba(99, 102, 241, 0.3)',
                color: '#ffffff',
                fontFamily: 'var(--font-mono)',
                fontSize: '15px',
                fontWeight: '600',
                outline: 'none',
                cursor: 'pointer'
              }}
            >
              {tickers.map((t) => (
                <option key={t} value={t}>
                  {t} {t === 'ATVI' ? '(Delisted 2023 - Historical Only)' : ''}
                </option>
              ))}
            </select>
          </div>

          {/* Model Architecture Selector */}
          <div>
            <label style={{ display: 'block', fontSize: '12px', color: '#94a3b8', marginBottom: '8px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Architecture
            </label>
            <div style={{ display: 'flex', gap: '8px', background: '#090d16', padding: '4px', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
              <button
                type="button"
                onClick={() => setModelType('gmu_fusion')}
                style={{
                  flex: 1,
                  padding: '9px 12px',
                  borderRadius: '6px',
                  border: 'none',
                  background: modelType === 'gmu_fusion' ? '#4f46e5' : 'transparent',
                  color: modelType === 'gmu_fusion' ? '#ffffff' : '#94a3b8',
                  fontSize: '12px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                GMU Fusion
              </button>
              <button
                type="button"
                onClick={() => setModelType('market_bilstm')}
                style={{
                  flex: 1,
                  padding: '9px 12px',
                  borderRadius: '6px',
                  border: 'none',
                  background: modelType === 'market_bilstm' ? '#4f46e5' : 'transparent',
                  color: modelType === 'market_bilstm' ? '#ffffff' : '#94a3b8',
                  fontSize: '12px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                Market BiLSTM
              </button>
            </div>
          </div>

          {/* Action Button */}
          <div>
            <button
              onClick={runPrediction}
              disabled={loading}
              className="btn-primary"
              style={{ width: '100%', padding: '13px 24px', fontSize: '14px', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px' }}
            >
              <Activity size={18} />
              {loading ? 'Executing Forward Inference...' : 'Evaluate Observation'}
            </button>
          </div>
        </div>
      </div>

      {/* Result Card */}
      {prediction && (
        <div className="glass-panel" style={{ padding: '32px', marginBottom: '28px', position: 'relative', overflow: 'hidden' }}>
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: '4px',
            background: prediction.predicted_class === 'Up' ? 'linear-gradient(90deg, #10b981, #34d399)' : 'linear-gradient(90deg, #ef4444, #f87171)'
          }} />

          {/* Header Row */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '20px', marginBottom: '24px' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span style={{ fontSize: '13px', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Forward Test Target Date: <strong style={{ color: '#f1f5f9' }}>{prediction.target_date}</strong>
                </span>
                {prediction.has_news ? (
                  <span className="badge badge-emerald" style={{ fontSize: '11px' }}>News Aligned</span>
                ) : (
                  <span className="badge badge-amber" style={{ fontSize: '11px' }}>Zero-News Fallback</span>
                )}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginTop: '10px' }}>
                <span style={{
                  fontSize: '40px',
                  fontWeight: '800',
                  color: prediction.predicted_class === 'Up' ? '#34d399' : '#f87171',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}>
                  {prediction.predicted_class === 'Up' ? <ArrowUpRight size={42} /> : <ArrowDownRight size={42} />}
                  {prediction.predicted_class}
                </span>
                <div>
                  <div style={{ fontSize: '12px', color: '#94a3b8' }}>Sigmoid Probability</div>
                  <div style={{ fontSize: '20px', fontWeight: '700', color: '#f8fafc', fontFamily: 'var(--font-mono)' }}>
                    {(prediction.probability * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>

            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '12px', color: '#94a3b8' }}>Ground Truth Direction</div>
              <div style={{ fontSize: '20px', fontWeight: '700', color: prediction.ground_truth_class === prediction.predicted_class ? '#34d399' : '#94a3b8', display: 'flex', alignItems: 'center', gap: '6px', justifyContent: 'flex-end', marginTop: '6px' }}>
                {prediction.ground_truth_class === prediction.predicted_class && <CheckCircle size={18} />}
                {prediction.ground_truth_class}
              </div>
              <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
                {prediction.ground_truth_class === prediction.predicted_class ? 'Prediction Correct' : 'Direction Misclassified'}
              </div>
            </div>
          </div>

          {/* Gating Mechanism Visualization (if GMU) */}
          {prediction.market_gate_weight !== undefined && prediction.news_gate_weight !== undefined && (
            <div style={{ background: 'rgba(15, 23, 42, 0.7)', padding: '20px', borderRadius: '12px', border: '1px solid rgba(255, 255, 255, 0.06)', marginBottom: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', fontWeight: 600, color: '#e2e8f0' }}>
                  <Sliders size={16} color="#818cf8" /> Learned GMU Gating Contribution
                </div>
                <div style={{ fontSize: '12px', color: '#94a3b8', fontFamily: 'var(--font-mono)' }}>
                  Market: <strong style={{ color: '#60a5fa' }}>{marketPct}%</strong> | News: <strong style={{ color: '#a78bfa' }}>{newsPct}%</strong>
                </div>
              </div>

              {/* Progress Bar */}
              <div style={{ height: '14px', width: '100%', background: 'rgba(0,0,0,0.4)', borderRadius: '7px', overflow: 'hidden', display: 'flex' }}>
                <div style={{ width: `${marketPct}%`, background: 'linear-gradient(90deg, #2563eb, #3b82f6)', transition: 'width 0.4s ease' }} title={`Market: ${marketPct}%`} />
                <div style={{ width: `${newsPct}%`, background: 'linear-gradient(90deg, #7c3aed, #8b5cf6)', transition: 'width 0.4s ease' }} title={`News: ${newsPct}%`} />
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#64748b', marginTop: '8px' }}>
                <span>Market Feature Representation (BiLSTM 64-dim)</span>
                <span>Financial News Representation (FinBERT 768-dim)</span>
              </div>
            </div>
          )}

          {/* Metadata Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', background: 'rgba(0, 0, 0, 0.25)', padding: '18px', borderRadius: '10px', marginBottom: '22px' }}>
            <div>
              <div style={{ fontSize: '11px', color: '#64748b' }}>Model Pipeline</div>
              <div style={{ fontSize: '13px', color: '#f8fafc', fontWeight: '600' }}>{prediction.model_version}</div>
            </div>
            <div>
              <div style={{ fontSize: '11px', color: '#64748b' }}>Modality Type</div>
              <div style={{ fontSize: '13px', color: '#f8fafc', fontWeight: '500' }}>{prediction.modality || 'Multimodal'}</div>
            </div>
            <div>
              <div style={{ fontSize: '11px', color: '#64748b' }}>Lookback Window</div>
              <div style={{ fontSize: '13px', color: '#f8fafc', fontWeight: '500' }}>{prediction.lookback_window} Market Sessions</div>
            </div>
            <div>
              <div style={{ fontSize: '11px', color: '#64748b' }}>Ticker Status</div>
              <div style={{ fontSize: '13px', color: '#f8fafc', fontWeight: '500' }}>{prediction.ticker_status || 'Active Equity'}</div>
            </div>
          </div>

          {/* Academic Disclaimer */}
          <div style={{ padding: '12px 16px', borderRadius: '8px', background: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.2)', display: 'flex', gap: '10px', alignItems: 'center' }}>
            <ShieldAlert size={18} color="#f87171" style={{ flexShrink: 0 }} />
            <div style={{ fontSize: '12px', color: '#fca5a5' }}>
              {prediction.disclaimer}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
