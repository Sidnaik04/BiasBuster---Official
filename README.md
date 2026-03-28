# 📚 Frontend Integration Files Index

**Date Generated:** March 28, 2026  
**Status:** ✅ Ready for Integration  
**Target:** Frontend Engineers (Next.js/React/TypeScript)

---

## 📂 Files Created for Your Team

### 1. **QUICK_START.md** (7.2 KB) ⭐ START HERE

**Read this first!** 5-minute quick reference guide.

**Contains:**

- 30-second setup overview
- 3 API endpoints reference
- Minimal code example
- Expected response formats
- Environment setup instructions
- Common tasks
- Testing checklist

**Time:** 5 minutes  
**Best for:** Getting started quickly

---

### 2. **BIAS_MITIGATION_INTEGRATION_GUIDE.md** (31 KB) 📖 FULL DOCUMENTATION

Complete, production-ready integration guide.

**Contains:**

- Overview of 3 features
- Complete API reference with examples
- Full source code for:
  - API service layer
  - React hooks (3x)
  - UI components (3x)
  - Error handling
  - State management (Zustand + Context)
- Best practices guide
- Complete page example
- Deployment checklist
- Troubleshooting guide

**Time:** 30-45 minutes  
**Best for:** Complete understanding and implementation

---

### 3. **lib/biasApi.ts** (9.6 KB) 💻 READY-TO-USE CODE

Production-ready API service implementation.

**Contains:**

- Type-safe API calls for all 3 endpoints
- Automatic timeout handling
- Comprehensive error handling
- Request validation utilities
- Health check function
- User-friendly error messages

**How to use:** Copy directly into your project!

---

### 4. **types/bias-mitigation.ts** (12 KB) 🔧 TYPESCRIPT TYPES

Complete TypeScript type definitions.

**Contains:**

- Request types for all endpoints
- Response types for all endpoints
- Fairness metrics types
- Performance metrics types
- Hook return types
- Component props types
- Utility type guards
- Helper functions (formatting, validation)

**How to use:** Import types in all your files

---

## 🎯 Integration Flow

```
┌──────────────────────────────────────────────┐
│ STEP 1: Read QUICK_START.md (5 min)          │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ STEP 2: Copy lib/biasApi.ts into your project│
│         Copy types/bias-mitigation.ts         │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ STEP 3: Create hooks from INTEGRATION_GUIDE  │
│         - useBiasDetection                   │
│         - useStrategyRecommendation          │
│         - useStrategyRanking                 │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ STEP 4: Create components from INTEGRATION   │
│         - BiasDetectionResults               │
│         - StrategyRecommendation             │
│         - StrategyRanking                    │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ STEP 5: Build your page using components    │
│         See INTEGRATION_GUIDE for example    │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ STEP 6: Test, customize styling, deploy     │
└──────────────────────────────────────────────┘
```

---

## 📋 Which File For What?

| Task                        | Read This                                                                 |
| --------------------------- | ------------------------------------------------------------------------- |
| Quick overview              | QUICK_START.md                                                            |
| Setup API service           | lib/biasApi.ts                                                            |
| Get TypeScript types        | types/bias-mitigation.ts                                                  |
| Create hooks                | BIAS_MITIGATION_INTEGRATION_GUIDE.md (Section: React Hooks)               |
| Build UI components         | BIAS_MITIGATION_INTEGRATION_GUIDE.md (Section: UI Components)             |
| Understand workflow         | BIAS_MITIGATION_INTEGRATION_GUIDE.md (Section: Integration Workflow)      |
| Error handling              | BIAS_MITIGATION_INTEGRATION_GUIDE.md (Section: Error Handling)            |
| State management            | BIAS_MITIGATION_INTEGRATION_GUIDE.md (Section: State Management)          |
| Full implementation example | BIAS_MITIGATION_INTEGRATION_GUIDE.md (Section: Examples)                  |
| Troubleshooting             | BIAS_MITIGATION_INTEGRATION_GUIDE.md (Section: Support & Troubleshooting) |

---

## 🎓 Learning Path

### Path A: Fast Track (1 hour)

1. Read QUICK_START.md (5 min)
2. Copy lib/biasApi.ts and types/bias-mitigation.ts (5 min)
3. Create one simple component (20 min)
4. Test on your local setup (20 min)
5. Get feedback from team (10 min)

### Path B: Thorough (2-3 hours)

1. Read QUICK_START.md (5 min)
2. Read full BIAS_MITIGATION_INTEGRATION_GUIDE.md (45 min)
3. Copy all code files (10 min)
4. Create all hooks and components (60 min)
5. Build complete page example (30 min)
6. Test thoroughly (10 min)

### Path C: Expert (Full Day)

Do Path B + Plus:

1. Study state management options (30 min)
2. Implement custom state solution (60 min)
3. Add advanced error handling (30 min)
4. Setup error logging (Sentry, etc.) (30 min)
5. Optimize performance (30 min)
6. Write tests for components (60 min)

---

## 🔗 API Endpoints Overview

### Endpoint 1: Detect Bias

```
POST /api/bias/detect
```

**Purpose:** Analyze dataset for bias  
**When:** Initially to understand bias in data

### Endpoint 2: Recommend Strategy

```
POST /api/bias/recommend-strategy
```

**Purpose:** Get pre-testing strategy hint  
**When:** Before running full tests

### Endpoint 3: Rank Strategies

```
POST /api/bias/rank-strategies
```

**Purpose:** Compare all three strategies after testing  
**When:** Determine which strategy to deploy

---

## 📦 Code Organization Example

```
frontend/
├── lib/
│   └── biasApi.ts                    ← You copy this
├── types/
│   └── bias-mitigation.ts            ← You copy this
├── hooks/
│   ├── useBiasDetection.ts           ← You create this
│   ├── useStrategyRecommendation.ts  ← You create this
│   └── useStrategyRanking.ts         ← You create this
├── components/
│   ├── BiasDetectionResults.tsx      ← You create this
│   ├── StrategyRecommendation.tsx    ← You create this
│   └── StrategyRanking.tsx           ← You create this
├── app/
│   └── bias-analysis/
│       └── page.tsx                  ← You create this
├── store/
│   └── biasStore.ts                  ← Optional: state management
├── QUICK_START.md                    ← You're reading this
└── BIAS_MITIGATION_INTEGRATION_GUIDE.md
```

---

## ⚡ Copy-Paste Commands

### Copy API Service

```bash
# Copy lib/biasApi.ts to your project
cp frontend/lib/biasApi.ts your-project/lib/

# Copy types
cp frontend/types/bias-mitigation.ts your-project/types/
```

### Create Hook Template

```typescript
// hooks/useBiasDetection.ts
import { useState } from "react";
import { biasApi } from "@/lib/biasApi";
import type { DetectBiasResponse } from "@/types/bias-mitigation";

export function useBiasDetection() {
  const [data, setData] = useState<DetectBiasResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

---

## ✅ Pre-Integration Checklist

- [ ] Backend running at `localhost:8000` (or configured in `.env.local`)
- [ ] Read QUICK_START.md
- [ ] Copy `lib/biasApi.ts` to your project
- [ ] Copy `types/bias-mitigation.ts` to your project
- [ ] Update `.env.local` with correct API URL
- [ ] Node modules installed (`npm install`)
- [ ] TypeScript configured properly
- [ ] React 18+ installed

---

## 🧪 Testing Your Integration

### 1. Test API Connection

```typescript
import { biasApi } from "@/lib/biasApi";

// In your component or test
const isHealthy = await biasApi.healthCheck();
console.log("API Connection:", isHealthy ? "✅" : "❌");
```

### 2. Test Bias Detection

```typescript
const response = await biasApi.detectBias({
  dataset_id: "test_dataset",
  model_id: "test_model",
  sensitive_attributes: ["gender"],
  target_column: "approved",
});
console.log("Bias detected:", response.bias_detected);
```

### 3. View in Component

```typescript
{data && <BiasDetectionResults data={data} />}
```

---

## 🚀 Deployment Steps

1. **Configure API URL**

   ```bash
   # .env.production
   NEXT_PUBLIC_API_URL=https://api.yourdomain.com/api
   ```

2. **Build & Test**

   ```bash
   npm run build
   npm run start
   ```

3. **Monitor Errors**
   - Add Sentry/LogRocket for error tracking
   - Monitor for timeout issues
   - Check API response times

4. **Optimize Performance**
   - Cache responses when appropriate
   - Implement request debouncing
   - Add progress indicators for long operations

---

## 💡 Pro Tips

1. **Start Simple** - Integrate bias detection first, then add others
2. **Test Locally** - Always test with backend running locally
3. **Handle Timeouts** - Strategy testing takes 30+ seconds, show spinner
4. **Cache Results** - Save responses to avoid re-running expensive operations
5. **Type Safety** - Use imported types for all API responses
6. **Error Messages** - Use `biasApiErrorHandling.getUserFriendlyMessage()` for UX

---

## 🆘 Common Issues

| Issue            | Solution                                    |
| ---------------- | ------------------------------------------- |
| API not found    | Check `NEXT_PUBLIC_API_URL` in `.env.local` |
| CORS error       | Backend needs CORS middleware               |
| Type errors      | Ensure types are imported correctly         |
| Timeout errors   | Increase timeout or show longer spinner     |
| JSON parse error | Check backend is returning valid JSON       |

---

## 📞 Need Help?

### For API Issues

→ See BIAS_MITIGATION_INTEGRATION_GUIDE.md (Section: "Support & Troubleshooting")

### For Backend Issues

→ See `backend/INDEX_TEST_FILES.md`

### For TypeScript Issues

→ Check `types/bias-mitigation.ts` for all type definitions

---

## 📚 External Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [React Hooks Documentation](https://react.dev/reference/react/hooks)
- [TypeScript Handbook](https://www.typescriptlang.org/docs)
- [Fetch API Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)

---

## ✨ Summary

You now have:

- ✅ Complete API service (`lib/biasApi.ts`)
- ✅ Full TypeScript types (`types/bias-mitigation.ts`)
- ✅ Step-by-step integration guide (BIAS_MITIGATION_INTEGRATION_GUIDE.md)
- ✅ Quick reference guide (QUICK_START.md)
- ✅ Production-ready code examples
- ✅ Error handling patterns
- ✅ State management options

**Everything you need to integrate bias mitigation into your frontend!**

---

## 🎯 Next Steps

1. **Start:** Read `QUICK_START.md` (5 minutes)
2. **Setup:** Copy `lib/biasApi.ts` and `types/bias-mitigation.ts`
3. **Create:** Make hooks and components from INTEGRATION_GUIDE
4. **Test:** Verify API connection and component rendering
5. **Deploy:** Push to production with confidence

---

**Version:** 1.0  
**Status:** ✅ Ready for Production  
**Last Updated:** March 28, 2026

Good luck with your integration! 🚀
