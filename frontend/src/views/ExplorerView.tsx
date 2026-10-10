import React, { useEffect, useState } from 'react';
import { fetchTickers, fetchHistory, TickerInfo } from '../api';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { TrendingUp, AlertTriangle, Calendar, Filter } from 'lucide-react';

export const ExplorerView: React.FC = () => {
  const [tickers, setTickers] = useState<TickerInfo[]>([]);
  const [selectedTicker, setSelectedTicker] = useState('AAPL');
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [showIndicators, setShowIndicators] = useState(true);

  useEffect(() => {
    fetchTickers().then((tList) => {
      setTickers(tList);
      loadHistory(selectedTicker);
    });
  }, []);

  const loadHistory = async (symbol: string) => {
    setLoading(true);
    setSelectedTicker(symbol);
    const data = await fetchHistory(symbol, 60);
    setHistory(data);
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: '1300px', margin: '0 auto', padding: '30px 20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h2 style={{ fontSize: '26px', fontWeight: '700', marginBottom: '6px' }}>Stock & Feature Explorer</h2>
          <p style={{ color: '#94a3b8', fontSize: '14px' }}>
            Inspect historical price actions, Bollinger Bands, EMAs, and RSI across the 12 equities.
          </p>
        </div>

        {/* Ticker Selector */}
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <select
            value={selectedTicker}
            onChange={(e) => loadHistory(e.target.value)}
            style={{
              padding: '10px 16px',
              borderRadius: '8px',
              background: '#0f172a',
              border: '1px solid rgba(99, 102, 241, 0.3)',
              color: '#ffffff',
              fontSize: '14px',
              fontFamily: 'var(--font-mono)',
              outline: 'none',
              cursor: 'pointer'
            }}
          >
            {tickers.map((t) => (
              <option key={t.symbol} value={t.symbol}>
                {t.symbol} - {t.name}
              </option>
            ))}
          </select>

          <button
            onClick={() => setShowIndicators(!showIndicators)}
            className="btn-outline"
            style={{ fontSize: '13px' }}
          >
            <Filter size={15} />
            {showIndicators ? 'Hide Indicators' : 'Show Indicators'}
          </button>
        </div>
      </div>

      {/* Main Chart Panel */}
      <div className="glass-panel" style={{ padding: '24px', marginBottom: '28px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div>
            <span style={{ fontSize: '18px', fontWeight: '700', color: '#f8fafc' }}>{selectedTicker} Price & Technical Bands</span>
            <span style={{ marginLeft: '12px', fontSize: '12px', color: '#94a3b8' }}>Latest 60 Trading Sessions</span>
          </div>
          {selectedTicker === 'ATVI' && (
            <span className="badge badge-amber" style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
              <AlertTriangle size={12} /> Delisted Historical Archive
            </span>
          )}
        </div>

        {loading ? (
          <div style={{ height: '360px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8' }}>
            Loading market data...
          </div>
        ) : history.length === 0 ? (
          <div style={{ height: '360px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8' }}>
            No local price data available for {selectedTicker}.
          </div>
        ) : (
          <div style={{ height: '380px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" />
                <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis domain={['auto', 'auto']} stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px', color: '#f8fafc' }} />
                <Line type="monotone" dataKey="close" stroke="#38bdf8" strokeWidth={2.5} dot={false} name="Close Price" />
                {showIndicators && (
                  <>
                    <Line type="monotone" dataKey="12-Day EMA" stroke="#818cf8" strokeWidth={1.5} dot={false} name="EMA 12" />
                    <Line type="monotone" dataKey="26-Day EMA" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="EMA 26" />
                    <Line type="monotone" dataKey="Upper Band" stroke="#ec4899" strokeDasharray="3 3" strokeWidth={1} dot={false} name="Upper BB" />
                    <Line type="monotone" dataKey="Lower Band" stroke="#ec4899" strokeDasharray="3 3" strokeWidth={1} dot={false} name="Lower BB" />
                  </>
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Feature Value Grid */}
      {history.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
          <div className="glass-panel" style={{ padding: '16px' }}>
            <div style={{ fontSize: '12px', color: '#94a3b8' }}>Latest Close</div>
            <div style={{ fontSize: '20px', fontWeight: '700', color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>
              ${Number(history[history.length - 1].close).toFixed(2)}
            </div>
          </div>
          <div className="glass-panel" style={{ padding: '16px' }}>
            <div style={{ fontSize: '12px', color: '#94a3b8' }}>RSI (14-Day)</div>
            <div style={{ fontSize: '20px', fontWeight: '700', color: '#f8fafc', fontFamily: 'var(--font-mono)' }}>
              {Number(history[history.length - 1].rsi || 50).toFixed(1)}
            </div>
          </div>
          <div className="glass-panel" style={{ padding: '16px' }}>
            <div style={{ fontSize: '12px', color: '#94a3b8' }}>MACD</div>
            <div style={{ fontSize: '20px', fontWeight: '700', color: '#818cf8', fontFamily: 'var(--font-mono)' }}>
              {Number(history[history.length - 1].MACD || 0).toFixed(3)}
            </div>
          </div>
          <div className="glass-panel" style={{ padding: '16px' }}>
            <div style={{ fontSize: '12px', color: '#94a3b8' }}>Slow %K</div>
            <div style={{ fontSize: '20px', fontWeight: '700', color: '#34d399', fontFamily: 'var(--font-mono)' }}>
              {Number(history[history.length - 1]['Slow k'] || 0).toFixed(1)}%
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
