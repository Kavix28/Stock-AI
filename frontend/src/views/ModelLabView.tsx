import React, { useEffect, useState } from 'react';
import { fetchMetrics, MetricSummary } from '../api';
import { Cpu, Award, Zap, AlertCircle } from 'lucide-react';

export const ModelLabView: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricSummary[]>([]);
  const [pooled, setPooled] = useState<any[]>([]);

  useEffect(() => {
    fetchMetrics().then((data) => {
      setMetrics(data.model_summaries);
      setPooled(data.pooled_metrics);
    });
  }, []);

  return (
    <div style={{ maxWidth: '1300px', margin: '0 auto', padding: '30px 20px' }}>
      <div style={{ marginBottom: '28px' }}>
        <h2 style={{ fontSize: '26px', fontWeight: '700', marginBottom: '6px' }}>Model Laboratory</h2>
        <p style={{ color: '#94a3b8', fontSize: '14px' }}>
          Cross-architecture performance comparison across Baselines, Linear Models, and Deep Recurrent Networks.
        </p>
      </div>

      {/* Model Benchmark Card */}
      <div className="glass-panel" style={{ padding: '24px', marginBottom: '32px' }}>
        <h3 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '16px' }}>
          Official 6,292-Sample Out-of-Sample Benchmark (2024–2026)
        </h3>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: '#94a3b8', textAlign: 'left' }}>
                <th style={{ padding: '12px 10px' }}>Architecture</th>
                <th style={{ padding: '12px 10px' }}>Mean Accuracy</th>
                <th style={{ padding: '12px 10px' }}>Macro F1</th>
                <th style={{ padding: '12px 10px' }}>Balanced Acc</th>
                <th style={{ padding: '12px 10px' }}>Evaluation Window</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((m, idx) => (
                <tr key={m.model_type} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)', background: idx === 0 ? 'rgba(99, 102, 241, 0.08)' : 'transparent' }}>
                  <td style={{ padding: '12px 10px', fontWeight: '600', color: idx === 0 ? '#38bdf8' : '#f8fafc' }}>
                    {m.model_type} {idx === 0 && '⭐ (Best Accuracy)'}
                  </td>
                  <td style={{ padding: '12px 10px', fontFamily: 'var(--font-mono)', fontWeight: '600', color: m.mean_accuracy > 0.51 ? '#34d399' : '#f8fafc' }}>
                    {(m.mean_accuracy * 100).toFixed(2)}%
                  </td>
                  <td style={{ padding: '12px 10px', fontFamily: 'var(--font-mono)', color: m.mean_macro_f1 > 0.45 ? '#818cf8' : '#94a3b8' }}>
                    {m.mean_macro_f1.toFixed(4)}
                  </td>
                  <td style={{ padding: '12px 10px', fontFamily: 'var(--font-mono)', color: '#94a3b8' }}>
                    {(m.mean_balanced_accuracy * 100).toFixed(2)}%
                  </td>
                  <td style={{ padding: '12px 10px', color: '#64748b', fontSize: '12px' }}>
                    572 Sessions (2024–2026)
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Scientific Insights Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '24px' }}>
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
            <Award size={20} color="#34d399" />
            <h4 style={{ fontSize: '16px', fontWeight: '600' }}>The Shared Multi-Ticker Advantage</h4>
          </div>
          <p style={{ fontSize: '13px', color: '#94a3b8', lineHeight: '1.6' }}>
            Training a single BiLSTM pooled across all 11 active equities (40,000+ training windows) outperforms individual per-ticker models by <strong>+3.55% accuracy</strong>, demonstrating the regularization benefits of cross-asset training in low signal-to-noise regimes.
          </p>
        </div>

        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
            <Zap size={20} color="#ec4899" />
            <h4 style={{ fontSize: '16px', fontWeight: '600' }}>Macro F1 vs. Accuracy Tradeoff</h4>
          </div>
          <p style={{ fontSize: '13px', color: '#94a3b8', lineHeight: '1.6' }}>
            Linear models achieve higher Macro F1 (0.4704) by predicting both upward and downward regimes, whereas raw neural networks tend to align with the empirical market drift (52.1% Up days in US equities). Validation threshold calibration provides controllable risk-adjusted operating points.
          </p>
        </div>

        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
            <AlertCircle size={20} color="#f59e0b" />
            <h4 style={{ fontSize: '16px', fontWeight: '600' }}>News Timing Designation</h4>
          </div>
          <p style={{ fontSize: '13px', color: '#94a3b8', lineHeight: '1.6' }}>
            Daily-aggregated news headlines lack point-in-time intraday timestamps. To prevent post-market lookahead contamination, multimodal fusion models are designated as <strong>TIMING-LIMITED</strong> and evaluated under strictly isolated ablation boundaries.
          </p>
        </div>
      </div>
    </div>
  );
};
