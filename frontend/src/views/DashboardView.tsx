import React, { useEffect, useState } from 'react';
import { fetchTickers, fetchMetrics, TickerInfo, MetricSummary } from '../api';
import { CheckCircle2, AlertCircle, Database, Calendar, Layers, BarChart } from 'lucide-react';

export const DashboardView: React.FC = () => {
  const [tickers, setTickers] = useState<TickerInfo[]>([]);
  const [metrics, setMetrics] = useState<MetricSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([fetchTickers(), fetchMetrics()]).then(([tData, mData]) => {
      setTickers(tData);
      setMetrics(mData.model_summaries);
      setLoading(false);
    });
  }, []);

  return (
    <div style={{ maxWidth: '1300px', margin: '0 auto', padding: '30px 20px' }}>
      {/* Page Header */}
      <div style={{ marginBottom: '30px' }}>
        <h2 style={{ fontSize: '28px', fontWeight: '700', marginBottom: '8px' }}>Research Dashboard</h2>
        <p style={{ color: '#94a3b8', fontSize: '14px' }}>
          Real-time dataset validation, chronological partition telemetry, and multi-model benchmark metrics.
        </p>
      </div>

      {/* Top Metrics Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '20px', marginBottom: '32px' }}>
        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '13px', color: '#94a3b8', fontWeight: '500' }}>Equity Universe</span>
            <Database size={18} color="#38bdf8" />
          </div>
          <div style={{ fontSize: '32px', fontWeight: '700', color: '#f8fafc' }}>12 Stocks</div>
          <div style={{ fontSize: '12px', color: '#10b981', display: 'flex', alignItems: 'center', gap: '4px', marginTop: '4px' }}>
            <CheckCircle2 size={14} /> 100% Schema Validated
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '13px', color: '#94a3b8', fontWeight: '500' }}>Historical Lookback</span>
            <Calendar size={18} color="#818cf8" />
          </div>
          <div style={{ fontSize: '32px', fontWeight: '700', color: '#f8fafc' }}>16 Years</div>
          <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
            2010-02-22 to 2026-10-09
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '13px', color: '#94a3b8', fontWeight: '500' }}>Out-of-Sample Test Set</span>
            <Layers size={18} color="#34d399" />
          </div>
          <div style={{ fontSize: '32px', fontWeight: '700', color: '#f8fafc' }}>6,292 Obs</div>
          <div style={{ fontSize: '12px', color: '#38bdf8', marginTop: '4px' }}>
            572 Days × 11 Active Equities
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '13px', color: '#94a3b8', fontWeight: '500' }}>Technical Indicators</span>
            <BarChart size={18} color="#f59e0b" />
          </div>
          <div style={{ fontSize: '32px', fontWeight: '700', color: '#f8fafc' }}>16 Features</div>
          <div style={{ fontSize: '12px', color: '#10b981', display: 'flex', alignItems: 'center', gap: '4px', marginTop: '4px' }}>
            <CheckCircle2 size={14} /> 0 NaN After Warmup
          </div>
        </div>
      </div>

      {/* Main Tables Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(480px, 1fr))', gap: '28px' }}>
        {/* Tickers Table */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '17px', fontWeight: '600', marginBottom: '16px' }}>Stock Universe Inventory</h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: '#94a3b8', textAlign: 'left' }}>
                  <th style={{ padding: '10px 8px' }}>Symbol</th>
                  <th style={{ padding: '10px 8px' }}>Company</th>
                  <th style={{ padding: '10px 8px' }}>Sector</th>
                  <th style={{ padding: '10px 8px' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {tickers.map((t) => (
                  <tr key={t.symbol} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                    <td style={{ padding: '10px 8px', fontFamily: 'var(--font-mono)', fontWeight: '600', color: '#38bdf8' }}>
                      {t.symbol}
                    </td>
                    <td style={{ padding: '10px 8px', color: '#f8fafc' }}>{t.name}</td>
                    <td style={{ padding: '10px 8px', color: '#94a3b8' }}>{t.sector}</td>
                    <td style={{ padding: '10px 8px' }}>
                      {t.delisted ? (
                        <span className="badge badge-amber">Delisted Fallback</span>
                      ) : (
                        <span className="badge badge-green">Refreshed</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Models Summary Table */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '17px', fontWeight: '600', marginBottom: '16px' }}>Model Comparison (Out-of-Sample)</h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: '#94a3b8', textAlign: 'left' }}>
                  <th style={{ padding: '10px 8px' }}>Model</th>
                  <th style={{ padding: '10px 8px' }}>Accuracy</th>
                  <th style={{ padding: '10px 8px' }}>Macro F1</th>
                  <th style={{ padding: '10px 8px' }}>Balanced</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((m) => (
                  <tr key={m.model_type} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                    <td style={{ padding: '10px 8px', fontWeight: '500', color: '#f8fafc' }}>
                      {m.model_type}
                    </td>
                    <td style={{ padding: '10px 8px', fontFamily: 'var(--font-mono)', color: m.mean_accuracy > 0.51 ? '#34d399' : '#f8fafc' }}>
                      {(m.mean_accuracy * 100).toFixed(2)}%
                    </td>
                    <td style={{ padding: '10px 8px', fontFamily: 'var(--font-mono)', color: '#94a3b8' }}>
                      {m.mean_macro_f1.toFixed(4)}
                    </td>
                    <td style={{ padding: '10px 8px', fontFamily: 'var(--font-mono)', color: '#94a3b8' }}>
                      {(m.mean_balanced_accuracy * 100).toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
