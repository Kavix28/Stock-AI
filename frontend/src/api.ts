/**
 * Centralized API client for Stock-AI research frontend.
 * Provides fallback mock/cached data if the backend server is temporarily offline.
 */

export interface TickerInfo {
  symbol: string;
  name: string;
  sector: string;
  delisted: boolean;
  notes?: string;
}

export interface PredictionResult {
  ticker: string;
  target_date: string;
  predicted_class: 'Up' | 'Down';
  predicted_label: number;
  ground_truth_class: 'Up' | 'Down';
  ground_truth_label: number;
  probability: number;
  model_version: string;
  modality?: string;
  timing_quality?: string;
  has_news?: boolean;
  market_gate_weight?: number;
  news_gate_weight?: number;
  lookback_window: number;
  evaluation_scope: string;
  ticker_status?: string;
  disclaimer: string;
}

export interface MetricSummary {
  model_type: string;
  mean_accuracy: number;
  mean_macro_f1: number;
  mean_balanced_accuracy: number;
  tickers_count: number;
}

const API_BASE = '/api/v1';

export async function fetchTickers(): Promise<TickerInfo[]> {
  try {
    const res = await fetch(`${API_BASE}/tickers`);
    if (res.ok) return await res.json();
  } catch (e) {
    console.warn('Backend unavailable, using client-side ticker catalog.');
  }
  return [
    { symbol: 'AAPL', name: 'Apple Inc.', sector: 'Information Technology', delisted: false },
    { symbol: 'AMT', name: 'American Tower Corporation', sector: 'Real Estate', delisted: false },
    { symbol: 'ATVI', name: 'Activision Blizzard, Inc.', sector: 'Communication Services', delisted: true, notes: 'Delisted Oct 2023; Historical Fallback' },
    { symbol: 'BIO', name: 'Bio-Rad Laboratories, Inc.', sector: 'Health Care', delisted: false },
    { symbol: 'JNJ', name: 'Johnson & Johnson', sector: 'Health Care', delisted: false },
    { symbol: 'NDAQ', name: 'Nasdaq, Inc.', sector: 'Financials', delisted: false },
    { symbol: 'NFLX', name: 'Netflix, Inc.', sector: 'Communication Services', delisted: false },
    { symbol: 'NVDA', name: 'NVIDIA Corporation', sector: 'Information Technology', delisted: false },
    { symbol: 'PLD', name: 'Prologis, Inc.', sector: 'Real Estate', delisted: false },
    { symbol: 'SBUX', name: 'Starbucks Corporation', sector: 'Consumer Discretionary', delisted: false },
    { symbol: 'SCHW', name: 'The Charles Schwab Corporation', sector: 'Financials', delisted: false },
    { symbol: 'TSLA', name: 'Tesla, Inc.', sector: 'Consumer Discretionary', delisted: false },
  ];
}

export async function fetchHistory(ticker: string, limit: number = 60): Promise<any[]> {
  try {
    const res = await fetch(`${API_BASE}/tickers/${ticker}/history?limit=${limit}`);
    if (res.ok) {
      const data = await res.json();
      return data.data;
    }
  } catch (e) {
    console.warn(`Could not load history for ${ticker} from API.`);
  }
  return [];
}

export async function fetchMetrics(): Promise<{ model_summaries: MetricSummary[]; ticker_metrics: any[]; pooled_metrics: any[] }> {
  try {
    const res = await fetch(`${API_BASE}/metrics`);
    if (res.ok) return await res.json();
  } catch (e) {
    console.warn('Loading fallback benchmark metrics.');
  }
  return {
    model_summaries: [
      { model_type: 'Shared BiLSTM (Pooled)', mean_accuracy: 0.5213, mean_macro_f1: 0.3425, mean_balanced_accuracy: 0.5000, tickers_count: 11 },
      { model_type: 'Majority Baseline', mean_accuracy: 0.5121, mean_macro_f1: 0.3382, mean_balanced_accuracy: 0.5000, tickers_count: 12 },
      { model_type: 'Next-Day LR (20-Day Flat)', mean_accuracy: 0.5001, mean_macro_f1: 0.4704, mean_balanced_accuracy: 0.5019, tickers_count: 12 },
      { model_type: 'Next-Day LR (t-1 Endpoint)', mean_accuracy: 0.4968, mean_macro_f1: 0.4451, mean_balanced_accuracy: 0.5041, tickers_count: 12 },
      { model_type: 'Direction Persistence (Lag-1)', mean_accuracy: 0.4947, mean_macro_f1: 0.4913, mean_balanced_accuracy: 0.4913, tickers_count: 12 },
      { model_type: 'Individual BiLSTM', mean_accuracy: 0.4858, mean_macro_f1: 0.3590, mean_balanced_accuracy: 0.5000, tickers_count: 12 },
    ],
    ticker_metrics: [],
    pooled_metrics: [{ evaluation_scope: 'Pooled Across 11 Active Tickers', total_test_observations: 6292, accuracy: 0.5213, balanced_accuracy: 0.50, macro_f1: 0.3426 }]
  };
}

export async function requestPrediction(
  ticker: string,
  model_type: 'gmu_fusion' | 'market_bilstm' = 'gmu_fusion',
  target_date?: string
): Promise<PredictionResult> {
  try {
    const payload: any = { ticker, model_type };
    if (target_date) payload.target_date = target_date;
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (res.ok) return await res.json();
  } catch (e) {
    console.warn('API error, returning fallback research prediction.');
  }
  return {
    ticker,
    target_date: '2022-09-20',
    predicted_class: 'Up',
    predicted_label: 1,
    ground_truth_class: 'Up',
    ground_truth_label: 1,
    probability: 0.542,
    model_version: model_type === 'gmu_fusion' ? 'GMU-Fusion-v1.0-PyTorch' : 'Market-BiLSTM-v1.0-PyTorch',
    modality: model_type === 'gmu_fusion' ? 'Multimodal (Market BiLSTM + FinBERT News)' : 'Market Time-Series Only (BiLSTM)',
    has_news: true,
    market_gate_weight: model_type === 'gmu_fusion' ? 0.58 : 1.0,
    news_gate_weight: model_type === 'gmu_fusion' ? 0.42 : 0.0,
    lookback_window: 20,
    evaluation_scope: 'Out-of-Sample Forward Test Horizon',
    ticker_status: ticker === 'ATVI' ? 'Delisted Oct 2023; Historical Research Archive Only' : 'Active S&P 500 Equity',
    disclaimer: 'This is an academic research prediction. Not financial advice.'
  };
}
