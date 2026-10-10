import React from 'react';
import { ArrowRight, Layers, ShieldCheck, Database, Award, BookOpen, Activity, AlertTriangle } from 'lucide-react';

interface LandingViewProps {
  onNavigate: (tab: string) => void;
}

export const LandingView: React.FC<LandingViewProps> = ({ onNavigate }) => {
  const tickers = ['AAPL', 'AMT', 'ATVI', 'BIO', 'JNJ', 'NDAQ', 'NFLX', 'NVDA', 'PLD', 'SBUX', 'SCHW', 'TSLA'];

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '40px 20px' }}>
      {/* Hero Section */}
      <div style={{ textAlign: 'center', marginBottom: '60px' }}>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', padding: '6px 14px', borderRadius: '20px', background: 'rgba(99, 102, 241, 0.1)', border: '1px solid rgba(99, 102, 241, 0.3)', marginBottom: '20px' }}>
          <ShieldCheck size={16} color="#38bdf8" />
          <span style={{ fontSize: '13px', color: '#a5b4fc', fontWeight: '500' }}>Rigorous, Leakage-Free Quantitative AI</span>
        </div>
        <h1 style={{ fontSize: '48px', fontWeight: '800', lineHeight: '1.15', marginBottom: '16px', background: 'linear-gradient(to right, #ffffff, #cbd5e1, #94a3b8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          Multimodal AI for Stock Trend Prediction
        </h1>
        <p style={{ fontSize: '18px', color: '#94a3b8', maxWidth: '750px', margin: '0 auto 30px', lineHeight: '1.6' }}>
          A reproducible, production-grade research platform investigating deep fusion of financial market time series, 16 technical indicators, and FinBERT sentiment representations across 12 equities.
        </p>

        <div style={{ display: 'flex', gap: '14px', justifyContent: 'center' }}>
          <button onClick={() => onNavigate('dashboard')} className="btn-primary" style={{ padding: '12px 24px', fontSize: '15px' }}>
            Open Research Dashboard <ArrowRight size={18} />
          </button>
          <button onClick={() => onNavigate('lab')} className="btn-outline" style={{ padding: '12px 24px', fontSize: '15px' }}>
            Inspect Benchmark Metrics
          </button>
        </div>
      </div>

      {/* Highlights Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '24px', marginBottom: '60px' }}>
        <div className="glass-panel" style={{ padding: '28px' }}>
          <div style={{ width: '42px', height: '42px', borderRadius: '10px', background: 'rgba(56, 189, 248, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '16px' }}>
            <Database size={22} color="#38bdf8" />
          </div>
          <h3 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '8px' }}>16-Year Curated Equities</h3>
          <p style={{ fontSize: '14px', color: '#94a3b8', lineHeight: '1.6' }}>
            4,185 daily trading sessions per active ticker spanning 2010 to 2026. Zero NaNs, strict OHLC boundary checks, and full indicator warmup isolation.
          </p>
        </div>

        <div className="glass-panel" style={{ padding: '28px' }}>
          <div style={{ width: '42px', height: '42px', borderRadius: '10px', background: 'rgba(99, 102, 241, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '16px' }}>
            <Layers size={22} color="#818cf8" />
          </div>
          <h3 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '8px' }}>Leakage-Free Windowing</h3>
          <p style={{ fontSize: '14px', color: '#94a3b8', lineHeight: '1.6' }}>
            20-day rolling sequence lookbacks strictly terminate at day <span style={{ fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>t-1</span>. MinMax normalization fit strictly on the training partition.
          </p>
        </div>

        <div className="glass-panel" style={{ padding: '28px' }}>
          <div style={{ width: '42px', height: '42px', borderRadius: '10px', background: 'rgba(16, 185, 129, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '16px' }}>
            <Award size={22} color="#34d399" />
          </div>
          <h3 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '8px' }}>Objective Benchmarking</h3>
          <p style={{ fontSize: '14px', color: '#94a3b8', lineHeight: '1.6' }}>
            Rigorous comparison against Training-Majority and Direction Persistence baselines over 6,292 untouched forward test observations (July 2024 – October 2026).
          </p>
        </div>
      </div>

      {/* Universe Chips */}
      <div className="glass-panel" style={{ padding: '28px', marginBottom: '40px', textAlign: 'center' }}>
        <h4 style={{ fontSize: '14px', textTransform: 'uppercase', color: '#94a3b8', letterSpacing: '1px', marginBottom: '16px' }}>
          Stock Universe (12 Benchmark Tickers)
        </h4>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', justifyContent: 'center' }}>
          {tickers.map((t) => (
            <span key={t} style={{
              padding: '6px 14px',
              borderRadius: '8px',
              background: t === 'ATVI' ? 'rgba(245, 158, 11, 0.15)' : 'rgba(30, 41, 59, 0.8)',
              border: `1px solid ${t === 'ATVI' ? 'rgba(245, 158, 11, 0.4)' : 'rgba(255, 255, 255, 0.1)'}`,
              fontFamily: 'var(--font-mono)',
              fontSize: '13px',
              fontWeight: '600',
              color: t === 'ATVI' ? '#fbbf24' : '#f8fafc'
            }}>
              {t} {t === 'ATVI' && '(Delisted Fallback)'}
            </span>
          ))}
        </div>
      </div>

      {/* Disclaimer Alert */}
      <div style={{ padding: '16px 20px', borderRadius: '10px', background: 'rgba(245, 158, 11, 0.08)', border: '1px solid rgba(245, 158, 11, 0.25)', display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
        <AlertTriangle size={20} color="#f59e0b" style={{ flexShrink: 0, marginTop: '2px' }} />
        <div style={{ fontSize: '13px', color: '#cbd5e1', lineHeight: '1.5' }}>
          <strong>Academic Research Notice:</strong> Stock-AI is designed strictly for scientific inquiry into market efficiency, point-in-time sequence representations, and multimodal fusion. It does not provide trading recommendations or investment advice.
        </div>
      </div>
    </div>
  );
};
