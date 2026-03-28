# 🚀 Quick Start - Bias Mitigation Frontend Integration

**File Location:** `frontend/BIAS_MITIGATION_INTEGRATION_GUIDE.md`  
**Target:** Frontend Engineers using Next.js/React/TypeScript  
**Time to read:** 5 minutes

---

## 📦 What You're Getting

Three production-ready features to integrate into your frontend:

```
┌─────────────────┐
│  BIAS DETECTION │ ← Analyze dataset/model for bias
└────────┬────────┘
         ↓
┌──────────────────────────┐
│ STRATEGY RECOMMENDER     │ ← Pre-test: which strategy to use?
└────────┬─────────────────┘
         ↓
┌──────────────────────────┐
│ STRATEGY RANKER          │ ← Post-test: rank all 3 strategies
└──────────────────────────┘
```

---

## ⚡ 30-Second Setup

### Step 1: Copy API Service

Create `lib/biasApi.ts` - Copy from Integration Guide (Section: "API Service Layer")

### Step 2: Create Hooks

Create files:

- `hooks/useBiasDetection.ts`
- `hooks/useStrategyRecommendation.ts`
- `hooks/useStrategyRanking.ts`

Copy code from Integration Guide (Section: "React Hook for...")

### Step 3: Add UI Components

Create files:

- `components/BiasDetectionResults.tsx`
- `components/StrategyRecommendation.tsx`
- `components/StrategyRanking.tsx`

Copy JSX from Integration Guide (Section: "UI Components")

---

## 🔗 API Endpoints

### 1. Detect Bias

```
POST /api/bias/detect
```

**When:** Initially analyze dataset for bias

### 2. Recommend Strategy

```
POST /api/bias/recommend-strategy
```

**When:** Get hint before testing all strategies

### 3. Rank Strategies

```
POST /api/bias/rank-strategies
```

**When:** After testing, determine best strategy

---

## 💻 Minimal Example

```typescript
// app/page.tsx
'use client';
import { useBiasDetection } from '@/hooks/useBiasDetection';
import { BiasDetectionResults } from '@/components/BiasDetectionResults';

export default function Home() {
  const { detectBias, data, loading, error } = useBiasDetection();

  return (
    <div className="p-6">
      <h1>Bias Mitigation</h1>

      <button onClick={() => detectBias('dataset1', 'model1', ['gender'], 'target')}>
        Detect Bias
      </button>

      {loading && <p>Analyzing...</p>}
      {error && <p className="text-red-600">{error}</p>}
      {data && <BiasDetectionResults data={data} />}
    </div>
  );
}
```

---

## 📊 Expected Response Format

### Bias Detection Response

```json
{
  "status": "success",
  "bias_detected": true,
  "bias_metrics": {
    "gender": {
      "dpd": 0.181,
      "eod": 0.109,
      "dir": 0.293,
      "performance": { "accuracy": 0.8537 }
    }
  }
}
```

### Recommendation Response

```json
{
  "recommended_strategy": "threshold",
  "confidence": 0.85,
  "reasoning": "Single binary attribute with high DPD..."
}
```

### Ranking Response

```json
{
  "best_strategy": "threshold",
  "best_score": 0.4369,
  "ranking": [{ "rank": 1, "strategy": "threshold", "total_score": 0.4369 }],
  "insights": ["🏆 THRESHOLD is a clear winner..."]
}
```

---

## ✅ Environment Setup

```bash
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

Or for production:

```bash
NEXT_PUBLIC_API_URL=https://api.yourdomain.com/api
```

---

## 🎨 Key Components Map

| Component                | Purpose               | Shows                             |
| ------------------------ | --------------------- | --------------------------------- |
| `BiasDetectionResults`   | Display bias analysis | DPD, EOD, DIR metrics             |
| `StrategyRecommendation` | Show pre-test hint    | Recommended strategy & confidence |
| `StrategyRanking`        | Display final ranking | All 3 strategies scored & ranked  |

---

## 🛠️ Common Tasks

### Detect Bias on Upload

```typescript
const { detectBias } = useBiasDetection();

async function handleUpload(file: File) {
  try {
    const result = await detectBias(datasetId, modelId, ["gender"], "target");
    if (result.bias_detected) {
      // Show recommendation
    }
  } catch (error) {
    // Handle error
  }
}
```

### Get Recommendation After Detection

```typescript
const { recommendStrategy } = useStrategyRecommendation();

const result = await recommendStrategy(metrics, ["gender"]);
console.log(`Recommended: ${result.recommended_strategy}`);
```

### Rank After Testing All Strategies

```typescript
const { rankStrategies } = useStrategyRanking();

const result = await rankStrategies(
  thresholdResults,
  reweightingResults,
  smoteResults,
);

console.log(`Deploy: ${result.best_strategy}`);
```

---

## 🔴 Error Handling Template

```typescript
try {
  const result = await detectBias(...);
} catch (error) {
  if (error instanceof TypeError) {
    // Network error
    setError('Connection failed');
  } else if (error instanceof SyntaxError) {
    // Invalid response
    setError('Invalid server response');
  } else {
    setError('Something went wrong');
  }
}
```

---

## 📱 Responsive Design Notes

- Results component: Stack vertically on mobile
- Metrics grid: 3 columns on desktop, 1 on mobile
- Strategy ranking: Full width cards

---

## 🧪 Testing Checklist

- [ ] Bias detection works end-to-end
- [ ] Shows results correctly
- [ ] Recommendation appears after bias detected
- [ ] Ranking displays all 3 strategies
- [ ] Error messages display properly
- [ ] Loading states show/hide correctly
- [ ] Mobile responsive
- [ ] Accessibility tested

---

## 📚 Full Documentation

See `BIAS_MITIGATION_INTEGRATION_GUIDE.md` for:

- Complete API reference (all parameters & responses)
- Full source code for all hooks & components
- State management examples (Zustand & Context)
- Advanced error handling patterns
- Best practices guide
- Complete page example
- Troubleshooting guide

---

## 🚀 Next Steps

1. **Read Full Guide** → `BIAS_MITIGATION_INTEGRATION_GUIDE.md`
2. **Copy Code** → Use provided hooks and components
3. **Test Locally** → Ensure backend is running at `localhost:8000`
4. **Customize** → Adjust styling and layout to match your design
5. **Deploy** → Set environment variables for production
6. **Monitor** → Add error logging (Sentry, LogRocket, etc.)

---

## 💡 Pro Tips

1. **Cache results** if user might go back-and-forth
2. **Show progress** - indicate which step is running (detection → recommendation → ranking)
3. **Explain metrics** - add tooltips for DPD, EOD, DIR
4. **Guide users** - show "next steps" after each stage
5. **Handle timeouts** - test might take 30+ seconds, show spinner
6. **Save recommendations** - user might want to compare later

---

## 🆘 Quick Troubleshooting

| Issue            | Solution                                    |
| ---------------- | ------------------------------------------- |
| API not found    | Check `NEXT_PUBLIC_API_URL` in `.env.local` |
| CORS error       | Backend needs CORS middleware enabled       |
| Timeout          | Mitigation testing takes 30+ seconds        |
| Invalid response | Ensure backend is running and healthy       |
| Build error      | Run `npm install` if missing dependencies   |

---

## 📞 Support

- Full Guide: `frontend/BIAS_MITIGATION_INTEGRATION_GUIDE.md`
- Backend Docs: `backend/INDEX_TEST_FILES.md`
- API Reference: `backend/README_TEST_GUIDE.md`

---

**Status:** ✅ Ready for Integration  
**Version:** 1.0  
**Generated:** March 28, 2026
