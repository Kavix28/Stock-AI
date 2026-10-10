import React, { useState } from 'react';
import { requestPrediction, PredictionResult } from '../api';
import { Activity, ArrowUpRight, ArrowDownRight, ShieldAlert, Sparkles, CheckCircle } from 'lucide-react';

export const PredictionDemoView: React.FC = () => {
  const tickers = ['AAPL', 'AMT', 'BIO', 'JNJ', 'NDAQ', 'NFLX', 'NVDA', 'PLD', 'SBUX', 'SCHW', 'TSLA', 'ATVI'];
  const [selectedTicker, setSelectedTicker] = useState('AAPL');
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);

  const runPrediction = async () => {
    setLoading(true);
    const res = await requestPrediction(selectedTicker);
    setPrediction(res);
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: '1000px', margin: '0 auto', padding: '30px 20px' }}>
      <div style={{ textAlign: 'center', marginBottom: '32px' }}>
        <h2 style={{ fontSize: '28px', fontWeight: '700', marginBottom: '8px' }}>Prediction Demonstration</h2>
        <p style={{ color: '#94a3b8', fontSize: '14px', maxWidth: '600px', margin: '0 auto' }}>
          Execute live model inference over verified test-set observation windows. Verified against historical ground truth without data fabrication.
        </p>
      </div>

      {/* Control Panel */}
      <div className="glass-panel" style={{ padding: '24px', marginBottom: '28px' }}>
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center', justifyContent: 'center', flexWrap: 'wrap' }}>
          <div>
            <label style={{ display: 'block', fontSize: '12px', color: '#94a3b8', marginBottom: '6px' }}>Select Stock Ticker</label>
            <select
              value={selectedTicker}
              onChange={(e) => setSelectedTicker(e.target.value)}
              style={{
                padding: '10px 20px',
                borderRadius: '8px',
                background: '#0f172a',
                border: '1px solid rgba(99, 102, 241, 0.4)',
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
                  {t} {t === 'ATVI' ? '(Historical Fallback)' : ''}
                </option>
              ))}
            </select>
          </div>

          <div style={{ alignSelf: 'flex-end' }}>
            <button
              onClick={runPrediction}
              disabled={loading}
              className="btn-primary"
              style={{ padding: '12px 28px', fontSize: '14px' }}
            >
              <Sparkles size={16} />
              {loading ? 'Evaluating Model...' : 'Execute Forecast'}
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

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '20px', marginBottom: '24px' }}>
            <div>
              <div style={{ fontSize: '13px', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Forecast for {prediction.ticker} (Target Date: {prediction.target_date})
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginTop: '8px' }}>
                <span style={{
                  fontSize: '36px',
                  fontWeight: '800',
                  color: prediction.predicted_class === 'Up' ? '#34d399' : '#f87171',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}>
                  {prediction.predicted_class === 'Up' ? <ArrowUpRight size={38} /> : <ArrowDownRight size={38} />}
                  {prediction.predicted_class}
                </span>
                <span className="badge badge-indigo" style={{ padding: '6px 12px', fontSize: '13px' }}>
                  {(prediction.probability * 100).toFixed(1)}% Confidence
                </span>
              </div>
            </div>

            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '12px', color: '#94a3b8' }}>Ground Truth Direction</div>
              <div style={{ fontSize: '18px', fontWeight: '700', color: prediction.ground_truth_class === prediction.predicted_class ? '#34d399' : '#94a3b8', display: 'flex', alignItems: 'center', gap: '4px', justifyContent: 'flex-end', marginTop: '4px' }}>
                {prediction.ground_truth_class === prediction.predicted_class && <CheckCircle size={16} />}
                {prediction.ground_truth_class}
              </div>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', background: 'rgba(0, 0, 0, 0.25)', padding: '16px', borderRadius: '8px', marginBottom: '20px' }}>
            <div>
              <div style={{ fontSize: '11px', color: '#64748b' }}>Model Architecture</div>
              <div style={{ fontSize: '13px', color: '#f8fafc', fontWeight: '500' }}>{prediction.model_version}</div>
            </div>
            <div>
              <div style={{ fontSize: '11px', color: '#64748b' }}>Market Lookback Window</div>
              <div style={{ fontSize: '13px', color: '#f8fafc', fontWeight: '500' }}>{prediction.lookback_window} Days [Ending t-1]</div>
            </div>
            <div>
              <div style={{ fontSize: '11px', color: '#64748b' }}>Evaluation Horizon</div>
              <div style={{ fontSize: '13px', color: '#f8fafc', fontWeight: '500' }}>{prediction.evaluation_scope}</div>
            </div>
          </div>

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
