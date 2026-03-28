# 🔧 Bias Mitigation Integration Guide - Frontend

**Target Audience:** Frontend Engineers  
**Tech Stack:** Next.js, TypeScript, React  
**API:** FastAPI Backend  
**Last Updated:** March 28, 2026

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [API Endpoints](#api-endpoints)
3. [Integration Workflow](#integration-workflow)
4. [API Reference](#api-reference)
5. [Frontend Implementation](#frontend-implementation)
6. [Error Handling](#error-handling)
7. [State Management](#state-management)
8. [UI Components](#ui-components)
9. [Best Practices](#best-practices)
10. [Examples](#examples)

---

## 🎯 Overview

The BiasBuster backend provides three interconnected features for bias mitigation:

### 1. **Bias Detection** 🔍

Detects and measures bias in your dataset and model predictions

**Endpoint:** `POST /api/bias/detect`  
**When to use:** Initial analysis to understand bias in your data

### 2. **Strategy Recommender** 💡

Suggests the best mitigation strategy BEFORE testing all three

**Endpoint:** `POST /api/bias/recommend-strategy`  
**When to use:** Quick guidance before running full mitigation tests

### 3. **Strategy Ranker** 🏆

Ranks all three strategies AFTER testing them on your data

**Endpoint:** `POST /api/bias/rank-strategies`  
**When to use:** Final decision on which strategy to deploy

---

## 🔗 API Endpoints

### 1. Detect Bias

```
POST /api/bias/detect
```

### 2. Recommend Strategy

```
POST /api/bias/recommend-strategy
```

### 3. Rank Strategies

```
POST /api/bias/rank-strategies
```

---

## 📊 Integration Workflow

```
┌─────────────────────────────────────────┐
│  User Uploads Dataset & Model            │
└─────────────────┬───────────────────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │ STEP 1: Detect Bias │
        └──────────┬──────────┘
                   │
        ✅ Bias detected? ──NO──→ Done
        │
        YES
        │
        ▼
    ┌──────────────────────────────┐
    │ STEP 2: Recommend Strategy   │
    │ (Optional - get quick hint)  │
    └──────────────┬───────────────┘
                   │
    (Show recommendation to user)
                   │
                   ▼
    ┌──────────────────────────────┐
    │ STEP 3: Apply & Test All 3   │
    │ • Threshold Optimizer        │
    │ • Reweighting                │
    │ • SMOTE                      │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │ STEP 4: Rank Strategies      │
    │ Get scores & comparison      │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │ Show Results & Insights      │
    │ Recommend best mitigation    │
    └──────────────────────────────┘
```

---

## 📡 API Reference

### Endpoint 1: Detect Bias

**Request:**

```json
{
  "dataset_id": "string (required)",
  "model_id": "string (required)",
  "sensitive_attributes": ["string (required)"],
  "target_column": "string (required)"
}
```

**Response:**

```json
{
  "status": "success",
  "bias_detected": true,
  "sensitive_attributes": ["gender", "age"],
  "bias_metrics": {
    "gender": {
      "dpd": 0.181,
      "eod": 0.109,
      "dir": 0.293,
      "performance": {
        "accuracy": 0.8537,
        "precision": 0.7369,
        "recall": 0.6043,
        "f1": 0.6641
      }
    }
  },
  "recommendation": "Bias detected in [gender]. Consider mitigation."
}
```

**Thresholds:**

- DPD > 0.10 = Bias detected
- EOD > 0.10 = Bias detected
- DIR < 0.80 = Bias detected

---

### Endpoint 2: Recommend Strategy (Pre-Testing)

**Request:**

```json
{
  "bias_metrics": {
    "gender": {
      "dpd": 0.181,
      "eod": 0.109,
      "dir": 0.293
    }
  },
  "sensitive_attributes": ["gender"]
}
```

**Response:**

```json
{
  "status": "success",
  "recommended_strategy": "threshold",
  "confidence": 0.85,
  "reasoning": "Single binary attribute with high DPD. Threshold Optimizer excels at this.",
  "alternatives": [
    {
      "strategy": "reweighting",
      "confidence": 0.15,
      "reason": "Works across attributes but slower convergence on single-attribute bias"
    }
  ],
  "next_step": "Run all three strategies to compare real results"
}
```

---

### Endpoint 3: Rank Strategies (Post-Testing)

**Request:**

```json
{
  "strategy_results": {
    "threshold": {
      "before_metrics": {
        "gender": {
          "dpd": 0.181,
          "eod": 0.109,
          "dir": 0.293,
          "performance": {
            "accuracy": 0.8537,
            "precision": 0.7369,
            "recall": 0.6043,
            "f1": 0.6641
          }
        }
      },
      "after_metrics": {
        "gender": {
          "dpd": 0.087,
          "eod": 0.036,
          "dir": 0.572,
          "performance": {
            "accuracy": 0.8338,
            "precision": 0.7107,
            "recall": 0.515,
            "f1": 0.5972
          }
        }
      }
    },
    "reweighting": {
      /* similar structure */
    },
    "smote": {
      /* similar structure */
    }
  }
}
```

**Response:**

```json
{
  "status": "success",
  "best_strategy": "threshold",
  "best_score": 0.4369,
  "ranking": [
    {
      "rank": 1,
      "strategy": "threshold",
      "total_score": 0.4369,
      "fairness_improvement": 0.4469,
      "accuracy_impact": -0.0199,
      "dpd_improvement": 0.0945,
      "eod_improvement": 0.0737,
      "di_improvement": 0.2787
    },
    {
      "rank": 2,
      "strategy": "reweighting",
      "total_score": 0.0484,
      "fairness_improvement": 0.0484,
      "accuracy_impact": 0.0,
      "dpd_improvement": 0.0072,
      "eod_improvement": 0.0265,
      "di_improvement": 0.0147
    },
    {
      "rank": 3,
      "strategy": "smote",
      "total_score": -0.2203,
      "fairness_improvement": -0.1988,
      "accuracy_impact": -0.043,
      "dpd_improvement": -0.1431,
      "eod_improvement": -0.0569,
      "di_improvement": -0.0049
    }
  ],
  "insights": [
    "🏆 THRESHOLD is a clear winner with 0.389 point lead over reweighting.",
    "📊 THRESHOLD sacrifices 2.0% accuracy for 0.546 fairness improvement.",
    "🎪 THRESHOLD improves all fairness metrics (DPD, EOD, DI)."
  ],
  "recommendation": "✅ Deploy Threshold Optimizer. It achieved the best fairness-accuracy balance."
}
```

---

## 🛠️ Frontend Implementation

### 1. API Service Layer

Create `lib/api.ts`:

```typescript
// lib/biasApi.ts
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

interface BiasMetrics {
  [attribute: string]: {
    dpd: number;
    eod: number;
    dir: number;
    performance?: {
      accuracy: number;
      precision: number;
      recall: number;
      f1: number;
    };
  };
}

interface DetectBiasRequest {
  dataset_id: string;
  model_id: string;
  sensitive_attributes: string[];
  target_column: string;
}

interface DetectBiasResponse {
  status: string;
  bias_detected: boolean;
  sensitive_attributes: string[];
  bias_metrics: BiasMetrics;
  recommendation: string;
}

interface RecommendStrategyRequest {
  bias_metrics: BiasMetrics;
  sensitive_attributes: string[];
}

interface RecommendStrategyResponse {
  status: string;
  recommended_strategy: "threshold" | "reweighting" | "smote";
  confidence: number;
  reasoning: string;
  alternatives: Array<{
    strategy: "threshold" | "reweighting" | "smote";
    confidence: number;
    reason: string;
  }>;
  next_step: string;
}

interface StrategyResult {
  before_metrics: BiasMetrics;
  after_metrics: BiasMetrics;
}

interface RankStrategiesRequest {
  strategy_results: {
    threshold: StrategyResult;
    reweighting: StrategyResult;
    smote: StrategyResult;
  };
}

interface RankStrategiesResponse {
  status: string;
  best_strategy: "threshold" | "reweighting" | "smote";
  best_score: number;
  ranking: Array<{
    rank: number;
    strategy: "threshold" | "reweighting" | "smote";
    total_score: number;
    fairness_improvement: number;
    accuracy_impact: number;
    dpd_improvement: number;
    eod_improvement: number;
    di_improvement: number;
  }>;
  insights: string[];
  recommendation: string;
}

// API Service Functions
export const biasApi = {
  async detectBias(request: DetectBiasRequest): Promise<DetectBiasResponse> {
    const response = await fetch(`${API_BASE}/bias/detect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!response.ok) throw new Error(`API Error: ${response.statusText}`);
    return response.json();
  },

  async recommendStrategy(
    request: RecommendStrategyRequest,
  ): Promise<RecommendStrategyResponse> {
    const response = await fetch(`${API_BASE}/bias/recommend-strategy`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!response.ok) throw new Error(`API Error: ${response.statusText}`);
    return response.json();
  },

  async rankStrategies(
    request: RankStrategiesRequest,
  ): Promise<RankStrategiesResponse> {
    const response = await fetch(`${API_BASE}/bias/rank-strategies`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!response.ok) throw new Error(`API Error: ${response.statusText}`);
    return response.json();
  },
};
```

### 2. React Hook for Bias Detection

```typescript
// hooks/useBiasDetection.ts
import { useState } from "react";
import { biasApi, DetectBiasResponse } from "@/lib/biasApi";

export function useBiasDetection() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<DetectBiasResponse | null>(null);

  const detectBias = async (
    datasetId: string,
    modelId: string,
    sensitiveAttrs: string[],
    targetCol: string,
  ) => {
    setLoading(true);
    setError(null);
    try {
      const response = await biasApi.detectBias({
        dataset_id: datasetId,
        model_id: modelId,
        sensitive_attributes: sensitiveAttrs,
        target_column: targetCol,
      });
      setData(response);
      return response;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { detectBias, data, loading, error };
}
```

### 3. React Hook for Strategy Recommendation

```typescript
// hooks/useStrategyRecommendation.ts
import { useState } from "react";
import { biasApi, RecommendStrategyResponse, BiasMetrics } from "@/lib/biasApi";

export function useStrategyRecommendation() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<RecommendStrategyResponse | null>(null);

  const recommendStrategy = async (
    metrics: BiasMetrics,
    attributes: string[],
  ) => {
    setLoading(true);
    setError(null);
    try {
      const response = await biasApi.recommendStrategy({
        bias_metrics: metrics,
        sensitive_attributes: attributes,
      });
      setData(response);
      return response;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { recommendStrategy, data, loading, error };
}
```

### 4. React Hook for Strategy Ranking

```typescript
// hooks/useStrategyRanking.ts
import { useState } from "react";
import { biasApi, RankStrategiesResponse, StrategyResult } from "@/lib/biasApi";

export function useStrategyRanking() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<RankStrategiesResponse | null>(null);

  const rankStrategies = async (
    thresholdResult: StrategyResult,
    reweightingResult: StrategyResult,
    smoteResult: StrategyResult,
  ) => {
    setLoading(true);
    setError(null);
    try {
      const response = await biasApi.rankStrategies({
        strategy_results: {
          threshold: thresholdResult,
          reweighting: reweightingResult,
          smote: smoteResult,
        },
      });
      setData(response);
      return response;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { rankStrategies, data, loading, error };
}
```

---

## 🎨 UI Components

### 1. Bias Detection Results Component

```typescript
// components/BiasDetectionResults.tsx
import { DetectBiasResponse } from '@/lib/biasApi';

interface Props {
  data: DetectBiasResponse;
}

export function BiasDetectionResults({ data }: Props) {
  if (!data.bias_detected) {
    return (
      <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
        <h3 className="text-lg font-semibold text-green-900">✅ No Bias Detected</h3>
        <p className="text-green-700 mt-2">Your model meets fairness criteria.</p>
      </div>
    );
  }

  return (
    <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
      <h3 className="text-lg font-semibold text-yellow-900">⚠️ Bias Detected</h3>

      {Object.entries(data.bias_metrics).map(([attribute, metrics]) => (
        <div key={attribute} className="mt-4 p-3 bg-white rounded border">
          <h4 className="font-medium text-gray-900">{attribute}</h4>
          <div className="grid grid-cols-3 gap-2 mt-2 text-sm">
            <div>
              <span className="text-gray-600">DPD:</span>
              <span className="ml-1 font-semibold">{metrics.dpd.toFixed(3)}</span>
            </div>
            <div>
              <span className="text-gray-600">EOD:</span>
              <span className="ml-1 font-semibold">{metrics.eod.toFixed(3)}</span>
            </div>
            <div>
              <span className="text-gray-600">DIR:</span>
              <span className="ml-1 font-semibold">{metrics.dir.toFixed(3)}</span>
            </div>
          </div>
          {metrics.performance && (
            <div className="mt-2 text-xs text-gray-600">
              Accuracy: {(metrics.performance.accuracy * 100).toFixed(2)}%
            </div>
          )}
        </div>
      ))}

      <p className="mt-4 text-yellow-900">{data.recommendation}</p>
    </div>
  );
}
```

### 2. Strategy Recommendation Component

```typescript
// components/StrategyRecommendation.tsx
import { RecommendStrategyResponse } from '@/lib/biasApi';

interface Props {
  data: RecommendStrategyResponse;
}

const strategyEmojis = {
  threshold: '🎯',
  reweighting: '⚖️',
  smote: '📊',
};

export function StrategyRecommendation({ data }: Props) {
  return (
    <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
      <div className="flex items-center gap-2">
        <span className="text-2xl">{strategyEmojis[data.recommended_strategy]}</span>
        <div>
          <h3 className="text-lg font-semibold text-blue-900">Recommended Strategy</h3>
          <p className="text-sm text-blue-700">{data.recommended_strategy.toUpperCase()}</p>
        </div>
        <div className="ml-auto px-3 py-1 bg-blue-600 text-white rounded-full text-sm font-medium">
          {(data.confidence * 100).toFixed(0)}% confidence
        </div>
      </div>

      <p className="mt-3 text-blue-900 text-sm">{data.reasoning}</p>

      {data.alternatives.length > 0 && (
        <div className="mt-4">
          <h4 className="font-medium text-blue-900 text-sm">Alternatives:</h4>
          <ul className="mt-2 space-y-1 text-xs text-blue-700">
            {data.alternatives.map((alt) => (
              <li key={alt.strategy}>
                <strong>{alt.strategy}:</strong> {alt.reason} ({(alt.confidence * 100).toFixed(0)}%)
              </li>
            ))}
          </ul>
        </div>
      )}

      <p className="mt-4 text-xs text-blue-600 italic">{data.next_step}</p>
    </div>
  );
}
```

### 3. Strategy Ranking Component

```typescript
// components/StrategyRanking.tsx
import { RankStrategiesResponse } from '@/lib/biasApi';

interface Props {
  data: RankStrategiesResponse;
}

export function StrategyRanking({ data }: Props) {
  const getRankColor = (rank: number) => {
    switch (rank) {
      case 1: return 'bg-yellow-50 border-yellow-200';
      case 2: return 'bg-gray-50 border-gray-200';
      case 3: return 'bg-orange-50 border-orange-200';
      default: return 'bg-white border-gray-200';
    }
  };

  const getRankEmoji = (rank: number) => {
    switch (rank) {
      case 1: return '🥇';
      case 2: return '🥈';
      case 3: return '🥉';
      default: return '•';
    }
  };

  return (
    <div className="space-y-4">
      <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
        <h3 className="text-lg font-semibold text-green-900">🏆 Best Strategy: {data.best_strategy.toUpperCase()}</h3>
        <p className="text-green-700 mt-2">{data.recommendation}</p>
      </div>

      <div className="space-y-2">
        {data.ranking.map((item) => (
          <div key={item.strategy} className={`p-4 border rounded-lg ${getRankColor(item.rank)}`}>
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                <span className="text-2xl">{getRankEmoji(item.rank)}</span>
                <div>
                  <h4 className="font-semibold text-gray-900 capitalize">{item.strategy}</h4>
                  <p className="text-xs text-gray-600 mt-1">
                    Fairness: {item.fairness_improvement.toFixed(4)} | Accuracy: {(item.accuracy_impact * 100).toFixed(2)}%
                  </p>
                </div>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-gray-900">{item.total_score.toFixed(4)}</div>
                <p className="text-xs text-gray-600">Score</p>
              </div>
            </div>

            <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
              <div className="bg-white rounded p-2">
                <span className="text-gray-600">DPD Improvement:</span>
                <div className="font-semibold text-gray-900">{item.dpd_improvement.toFixed(4)}</div>
              </div>
              <div className="bg-white rounded p-2">
                <span className="text-gray-600">EOD Improvement:</span>
                <div className="font-semibold text-gray-900">{item.eod_improvement.toFixed(4)}</div>
              </div>
              <div className="bg-white rounded p-2">
                <span className="text-gray-600">DI Improvement:</span>
                <div className="font-semibold text-gray-900">{item.di_improvement.toFixed(4)}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {data.insights.length > 0 && (
        <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <h4 className="font-semibold text-blue-900 mb-2">💡 Insights</h4>
          <ul className="space-y-1 text-sm text-blue-800">
            {data.insights.map((insight, idx) => (
              <li key={idx}>{insight}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
```

---

## ❌ Error Handling

```typescript
// hooks/useBiasDetection.ts with error handling
export function useBiasDetection() {
  // ... state setup ...

  const detectBias = async (...) => {
    try {
      // validation
      if (!datasetId || !modelId) {
        throw new Error('Dataset ID and Model ID are required');
      }

      const response = await biasApi.detectBias({...});
      setData(response);
      return response;

    } catch (err) {
      let message = 'Unknown error occurred';

      if (err instanceof Response) {
        if (err.status === 404) message = 'Dataset or model not found';
        else if (err.status === 400) message = 'Invalid request parameters';
        else if (err.status === 500) message = 'Server error';
      } else if (err instanceof Error) {
        message = err.message;
      }

      setError(message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { detectBias, data, loading, error };
}
```

---

## 💾 State Management

### Option 1: Zustand Store

```typescript
// store/biasStore.ts
import { create } from "zustand";
import {
  DetectBiasResponse,
  RecommendStrategyResponse,
  RankStrategiesResponse,
} from "@/lib/biasApi";

interface BiasStore {
  // Bias Detection
  biasDetection: DetectBiasResponse | null;
  setBiasDetection: (data: DetectBiasResponse) => void;

  // Strategy Recommendation
  recommendation: RecommendStrategyResponse | null;
  setRecommendation: (data: RecommendStrategyResponse) => void;

  // Strategy Ranking
  ranking: RankStrategiesResponse | null;
  setRanking: (data: RankStrategiesResponse) => void;

  // Loading & Error states
  loading: boolean;
  setLoading: (loading: boolean) => void;
  error: string | null;
  setError: (error: string | null) => void;

  // Reset
  reset: () => void;
}

export const useBiasStore = create<BiasStore>((set) => ({
  biasDetection: null,
  setBiasDetection: (data) => set({ biasDetection: data }),

  recommendation: null,
  setRecommendation: (data) => set({ recommendation: data }),

  ranking: null,
  setRanking: (data) => set({ ranking: data }),

  loading: false,
  setLoading: (loading) => set({ loading }),

  error: null,
  setError: (error) => set({ error }),

  reset: () =>
    set({
      biasDetection: null,
      recommendation: null,
      ranking: null,
      error: null,
      loading: false,
    }),
}));
```

### Option 2: React Context

```typescript
// context/BiasContext.tsx
import React, { createContext, useState, useContext } from 'react';
import { DetectBiasResponse, RecommendStrategyResponse, RankStrategiesResponse } from '@/lib/biasApi';

interface BiasContextType {
  biasDetection: DetectBiasResponse | null;
  recommendation: RecommendStrategyResponse | null;
  ranking: RankStrategiesResponse | null;
  loading: boolean;
  error: string | null;
  setBiasDetection: (data: DetectBiasResponse) => void;
  setRecommendation: (data: RecommendStrategyResponse) => void;
  setRanking: (data: RankStrategiesResponse) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

const BiasContext = createContext<BiasContextType | undefined>(undefined);

export function BiasProvider({ children }: { children: React.ReactNode }) {
  const [biasDetection, setBiasDetection] = useState<DetectBiasResponse | null>(null);
  const [recommendation, setRecommendation] = useState<RecommendStrategyResponse | null>(null);
  const [ranking, setRanking] = useState<RankStrategiesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reset = () => {
    setBiasDetection(null);
    setRecommendation(null);
    setRanking(null);
    setError(null);
    setLoading(false);
  };

  return (
    <BiasContext.Provider value={{
      biasDetection,
      recommendation,
      ranking,
      loading,
      error,
      setBiasDetection,
      setRecommendation,
      setRanking,
      setLoading,
      setError,
      reset,
    }}>
      {children}
    </BiasContext.Provider>
  );
}

export function useBias() {
  const context = useContext(BiasContext);
  if (!context) throw new Error('useBias must be used within BiasProvider');
  return context;
}
```

---

## 🔧 Best Practices

### 1. Always Show Loading States

```typescript
{loading && <Spinner />}
```

### 2. Display Error Messages Clearly

```typescript
{error && (
  <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
    <p className="text-red-900">❌ Error: {error}</p>
  </div>
)}
```

### 3. Validate Input Before API Calls

```typescript
if (!datasetId.trim()) {
  setError("Dataset ID cannot be empty");
  return;
}
```

### 4. Handle Missing Data Gracefully

```typescript
if (!data?.bias_metrics) {
  return <div>No bias data available</div>;
}
```

### 5. Provide Clear User Feedback

- Show which step is being executed
- Display progress indicators
- Explain what results mean
- Provide actionable recommendations

### 6. Cache Results When Possible

```typescript
const cachedResults = useRef<RankStrategiesResponse | null>(null);

const rankStrategies = async (...) => {
  if (cachedResults.current) {
    return cachedResults.current;
  }
  // ... make API call
  cachedResults.current = response;
};
```

---

## 📚 Examples

### Complete Page Example

```typescript
// app/bias-analysis/page.tsx
'use client';

import { useState } from 'react';
import { useBiasDetection } from '@/hooks/useBiasDetection';
import { useStrategyRecommendation } from '@/hooks/useStrategyRecommendation';
import { useStrategyRanking } from '@/hooks/useStrategyRanking';
import { BiasDetectionResults } from '@/components/BiasDetectionResults';
import { StrategyRecommendation } from '@/components/StrategyRecommendation';
import { StrategyRanking } from '@/components/StrategyRanking';

export default function BiasAnalysisPage() {
  const [datasetId, setDatasetId] = useState('');
  const [modelId, setModelId] = useState('');
  const [step, setStep] = useState<'initial' | 'detection' | 'recommendation' | 'ranking'>('initial');

  const { detectBias, data: biasData, loading: detectLoading, error: detectError } = useBiasDetection();
  const { recommendStrategy, data: recData, loading: recLoading } = useStrategyRecommendation();
  const { rankStrategies, data: rankData, loading: rankLoading } = useStrategyRanking();

  const handleDetectBias = async () => {
    try {
      const result = await detectBias(datasetId, modelId, ['gender', 'age'], 'target');
      setStep('detection');
      if (result.bias_detected) {
        setStep('recommendation');
      }
    } catch (err) {
      console.error('Detection failed:', err);
    }
  };

  const handleRecommendStrategy = async () => {
    if (!biasData?.bias_metrics) return;
    try {
      await recommendStrategy(biasData.bias_metrics, biasData.sensitive_attributes);
      setStep('ranking');
    } catch (err) {
      console.error('Recommendation failed:', err);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">🔍 Bias Analysis & Mitigation</h1>

      {step === 'initial' && (
        <div className="p-4 border rounded-lg">
          <h2 className="text-xl font-semibold mb-4">Step 1: Detect Bias</h2>
          <div className="space-y-2">
            <input
              type="text"
              placeholder="Dataset ID"
              value={datasetId}
              onChange={(e) => setDatasetId(e.target.value)}
              className="w-full p-2 border rounded"
            />
            <input
              type="text"
              placeholder="Model ID"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              className="w-full p-2 border rounded"
            />
            <button
              onClick={handleDetectBias}
              disabled={detectLoading}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
            >
              {detectLoading ? 'Analyzing...' : 'Analyze for Bias'}
            </button>
          </div>
          {detectError && <p className="mt-2 text-red-600">{detectError}</p>}
        </div>
      )}

      {step === 'detection' && biasData && (
        <div className="space-y-4">
          <BiasDetectionResults data={biasData} />
          {biasData.bias_detected && (
            <button
              onClick={handleRecommendStrategy}
              disabled={recLoading}
              className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
            >
              {recLoading ? 'Recommending...' : 'Continue to Strategy Recommendation'}
            </button>
          )}
        </div>
      )}

      {step === 'recommendation' && recData && (
        <div className="space-y-4">
          <StrategyRecommendation data={recData} />
          <button
            onClick={handleRecommendStrategy}
            className="px-4 py-2 bg-purple-600 text-white rounded"
          >
            Proceed to Strategy Ranking
          </button>
        </div>
      )}

      {step === 'ranking' && rankData && (
        <StrategyRanking data={rankData} />
      )}
    </div>
  );
}
```

---

## 🚀 Deployment Checklist

- [ ] Set `NEXT_PUBLIC_API_URL` in `.env.local`
- [ ] Update API endpoints if running on different server
- [ ] Test all three workflows end-to-end
- [ ] Add loading states to all async operations
- [ ] Implement error boundaries
- [ ] Add analytics tracking (optional)
- [ ] Set up error logging (Sentry, LogRocket, etc.)
- [ ] Test on mobile and desktop
- [ ] Verify CORS headers from backend
- [ ] Add authentication if needed

---

## 📞 Support & Troubleshooting

### API Connection Issues

```typescript
// Add to lib/biasApi.ts
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

if (typeof window !== "undefined") {
  console.log("API Base URL:", API_BASE);
}
```

### CORS Errors

Ensure backend has CORS enabled:

```python
# In FastAPI backend
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "your-domain.com"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)
```

### Request Timeout

```typescript
const response = await fetch(`${API_BASE}/bias/detect`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(request),
  signal: AbortSignal.timeout(30000), // 30 second timeout
});
```

---

## 📖 Additional Resources

- [Backend API Documentation](./INDEX_TEST_FILES.md)
- [Test Results Summary](./TEST_RESULTS_SUMMARY.md)
- [Strategy Evaluator Documentation](./README_TEST_GUIDE.md)

---

## ✅ Summary

**Three-Step Integration:**

1. **Setup API Service** → Use `biasApi` functions to call backend endpoints
2. **Create React Hooks** → Manage loading, error, and data states
3. **Build UI Components** → Display results and guide user through workflow

**Key Points:**

- Always handle loading and error states
- Validate user input before API calls
- Display clear, actionable feedback
- Use state management for complex workflows
- Test thoroughly before deploying

---

**Generated:** March 28, 2026  
**Version:** 1.0  
**Status:** ✅ Production Ready
