# A/B Test Plan: Restaurant Recommendations

**Experiment Name:** ML-Powered Restaurant Recommendations V1  
**Owner:** Ayush Saxena  
**Date:** January 2026  
**Status:** Ready for Launch  

---

## 1. Hypothesis

### Primary Hypothesis
**H1:** Users with ML-powered personalized recommendations will have 40% lower time-to-order compared to users with the current "Sort by: Distance/Rating" approach.

**Null Hypothesis (H0):** There is no significant difference in time-to-order between treatment and control groups.

**Alternative Hypothesis (H1):** Treatment group has significantly lower time-to-order (one-tailed test).

### Secondary Hypotheses
**H2:** Treatment group will have 15% higher order conversion rate  
**H3:** Treatment group will order from 2+ new restaurants per month  
**H4:** Treatment group will have higher user satisfaction scores  

---

## 2. Success Metrics

### 2.1 Primary Metric

| Metric | Baseline | Target | Minimum Detectable Effect |
|--------|----------|--------|---------------------------|
| **Time to Order** | 10 minutes | 6 minutes | 2 minutes (20% reduction) |

**Measurement:**
```python
time_to_order = order_placed_timestamp - home_page_load_timestamp

# Only count sessions where order was placed
# Exclude sessions >30 minutes (outliers)
```

### 2.2 Secondary Metrics

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Order Conversion Rate | 12% | 13.8% (+15%) | Orders / Sessions |
| Discovery Rate | 1.2/month | 2.0/month | New restaurants ordered |
| Repeat Order Rate | 65% | 55% (-10%) | Orders from previous restaurants |
| User Satisfaction | 4.1/5 | 4.3/5 | Post-order survey rating |

### 2.3 Guardrail Metrics (Must Not Degrade)

| Metric | Baseline | Threshold | Action if Breached |
|--------|----------|-----------|-------------------|
| Avg Delivery Time | 35 min | 38 min | Stop test |
| Cancellation Rate | 5% | 6% | Investigate |
| Negative Feedback | 8% | 10% | Investigate |
| Revenue per Order | ₹380 | ₹350 | Monitor |

---

## 3. Experiment Design

### 3.1 Test Groups

**Control Group (50%):**
- Current experience: Sort by Distance/Rating/Delivery Time
- No personalization
- Generic restaurant listing

**Treatment Group (50%):**
- ML-powered personalized recommendations
- Top-10 highlighted section at top of feed
- Explanations for each recommendation
- Context-aware (time, weather, location)

### 3.2 Randomization

**Unit of Randomization:** User ID

**Randomization Method:**
```python
def assign_group(user_id):
    # Deterministic hash-based assignment
    hash_value = hashlib.md5(user_id.encode()).hexdigest()
    hash_int = int(hash_value, 16)
    
    if hash_int % 2 == 0:
        return "control"
    else:
        return "treatment"
```

**Why Hash-Based?**
- Deterministic: Same user always gets same experience
- Balanced: ~50-50 split
- No bias: Independent of user attributes

### 3.3 Sample Size Calculation

**Parameters:**
- Baseline time-to-order: 10 minutes (σ = 4 minutes)
- Minimum detectable effect: 2 minutes (20% reduction)
- Significance level (α): 0.05
- Statistical power (1-β): 0.80

**Calculation:**
```python
from scipy.stats import ttest_ind_from_stats
import numpy as np

# Effect size
effect_size = 2 / 4  # 0.5 (Cohen's d)

# Sample size per group
n_per_group = (2 * (1.96 + 0.84)**2) / (effect_size**2)
n_per_group = 64

# Safety margin (1.5×)
n_per_group_safe = 64 * 1.5 = 96

# Total users needed
total_users = 96 * 2 = 192
```

**Decision:** Test with 10,000 users per group (safety margin for multiple metrics)

### 3.4 Test Duration

**Calculation:**
```
Daily active users: 5,000
Users per group: 2,500 (50%)
Required sample: 10,000 per group

Days needed = 10,000 / 2,500 = 4 days
```

**Actual Duration:** 14 days

**Why 14 days?**
- Capture full week (weekday + weekend behavior)
- Account for day-of-week effects
- Allow for metric stabilization
- Detect potential novelty effects

---

## 4. Implementation

### 4.1 Instrumentation

**Events to Track:**
```python
# Page load
track_event("home_page_loaded", {
    "user_id": user_id,
    "timestamp": timestamp,
    "test_group": "treatment",  # or "control"
    "context": {
        "time_of_day": "dinner",
        "weather": "clear",
        "location": (lat, lon)
    }
})

# Recommendation displayed
track_event("recommendations_shown", {
    "user_id": user_id,
    "timestamp": timestamp,
    "restaurant_ids": [list_of_10],
    "model_scores": [list_of_scores]
})

# Restaurant clicked
track_event("restaurant_clicked", {
    "user_id": user_id,
    "timestamp": timestamp,
    "restaurant_id": restaurant_id,
    "rank": 3,  # Position in list
    "explanation_viewed": True
})

# Order placed
track_event("order_placed", {
    "user_id": user_id,
    "timestamp": timestamp,
    "restaurant_id": restaurant_id,
    "order_value": 450,
    "from_recommendation": True,
    "recommendation_rank": 3
})
```

### 4.2 Feature Flags
```python
# Feature flag configuration
feature_flags = {
    "ml_recommendations": {
        "enabled": True,
        "rollout_percentage": 50,
        "whitelist_users": [],  # VIP users get treatment
        "blacklist_users": []   # Exclude if issues
    }
}

# Usage
if is_enabled("ml_recommendations", user_id):
    recommendations = ml_model.recommend(user_id)
else:
    recommendations = default_sort_by_rating()
```

### 4.3 Gradual Rollout

**Phase 1: Internal (Week 1)**
- 100 internal employees
- Goal: Catch critical bugs
- Success criteria: No crashes, latency <2s

**Phase 2: Alpha (Week 2)**
- 1% of users (5,000)
- Goal: Validate instrumentation
- Success criteria: Data logging works

**Phase 3: Beta (Week 3-4)**
- 10% of users (50,000)
- Goal: Detect any adverse effects
- Success criteria: Guardrails not breached

**Phase 4: Full A/B Test (Week 5-6)**
- 50% of users (treatment group)
- Goal: Measure impact on key metrics
- Success criteria: Primary metric improved

**Phase 5: Full Launch (Week 7+)**
- 100% of users
- Goal: Production deployment
- Success criteria: Sustained improvement

---

## 5. Analysis Plan

### 5.1 Statistical Tests

**Primary Metric (Time to Order):**
- **Test:** Two-sample t-test (one-tailed)
- **Significance level:** α = 0.05
- **Effect direction:** Treatment < Control
```python
from scipy.stats import ttest_ind

control_times = df[df['group'] == 'control']['time_to_order']
treatment_times = df[df['group'] == 'treatment']['time_to_order']

t_stat, p_value = ttest_ind(control_times, treatment_times, alternative='less')

if p_value < 0.05 and treatment_mean < control_mean:
    conclusion = "Reject null hypothesis: Treatment is significantly better"
else:
    conclusion = "Fail to reject null hypothesis"
```

**Secondary Metrics:**
- Conversion rate: Two-proportion z-test
- Discovery rate: Two-sample t-test
- Satisfaction: Mann-Whitney U test (ordinal data)

### 5.2 Multiple Comparison Correction

**Problem:** Testing multiple metrics increases false positive risk

**Solution:** Bonferroni correction
```python
# 1 primary + 4 secondary = 5 tests
corrected_alpha = 0.05 / 5 = 0.01

# Use α = 0.01 for statistical significance
```

### 5.3 Segmentation Analysis

**Subgroup Analysis:**
1. **By User Segment:**
   - New users (<3 orders)
   - Regular users (3-20 orders)
   - Power users (20+ orders)

2. **By Time of Day:**
   - Breakfast, Lunch, Dinner, Late night

3. **By City:**
   - If multi-city launch
```python
# Example: Segment analysis
for segment in ['new_users', 'regular_users', 'power_users']:
    segment_data = df[df['user_segment'] == segment]
    
    control = segment_data[segment_data['group'] == 'control']
    treatment = segment_data[segment_data['group'] == 'treatment']
    
    effect = treatment['time_to_order'].mean() - control['time_to_order'].mean()
    print(f"{segment}: {effect:.2f} minutes reduction")
```

### 5.4 Sensitivity Analysis

**Check for:**
1. **Novelty Effect:** Does effect diminish over time?
```python
   # Compare Week 1 vs Week 2
   week1_effect = calculate_effect(df[df['week'] == 1])
   week2_effect = calculate_effect(df[df['week'] == 2])
```

2. **Simpson's Paradox:** Is effect consistent across segments?

3. **Outliers:** Remove top/bottom 1% and re-test

---

## 6. Decision Criteria

### 6.1 Launch Decision Matrix

| Condition | Primary Metric | Guardrails | Decision |
|-----------|----------------|------------|----------|
| ✅ Success | Improved ≥20% | Not breached | **LAUNCH** |
| ⚠️ Partial | Improved 10-20% | Not breached | Launch with monitoring |
| ❌ Neutral | Improved <10% | Not breached | Do not launch |
| 🛑 Failure | Any | Breached | Stop immediately |

### 6.2 Launch Criteria (All Must Be Met)

✅ **Statistical Significance:** p < 0.05 on primary metric  
✅ **Practical Significance:** ≥20% improvement on time-to-order  
✅ **Guardrail Metrics:** None breached  
✅ **User Feedback:** Net Promoter Score ≥ baseline  
✅ **Technical Stability:** Latency <2s, Error rate <1%  

### 6.3 Rollback Triggers

**Automatic Rollback If:**
- Error rate >5% for 15 minutes
- Latency p95 >5 seconds for 30 minutes
- Order cancellation rate >10%

**Manual Rollback If:**
- Guardrail metrics breached
- Significant negative user feedback
- Business stakeholder concern

---

## 7. Reporting

### 7.1 Daily Dashboard

**Metrics to Monitor:**
```
┌─────────────────────────────────────────────────────┐
│  Restaurant Recommendations A/B Test - Day 7/14     │
├─────────────────────────────────────────────────────┤
│  PRIMARY METRIC                                     │
│  Time to Order                                      │
│    Control:    10.2 min  (n=8,523)                  │
│    Treatment:   7.8 min  (n=8,491)                  │
│    Δ:          -2.4 min  (-23.5%)  ✅                │
│    p-value:     0.003    ✅                          │
│                                                     │
│  SECONDARY METRICS                                  │
│  Conversion Rate:   12.1% → 13.8%  (+14%)  ✅        │
│  Discovery Rate:     1.3 → 1.8     (+38%)  ✅        │
│                                                     │
│  GUARDRAILS                                         │
│  Delivery Time:     35.2 → 36.1    (+0.9)  ✅        │
│  Cancellation:      5.1% → 5.3%  (+0.2%)   ✅        │
│                                                     │
│  TECHNICAL METRICS                                  │
│  Latency (p95):      1.8s                  ✅        │
│  Error Rate:         0.3%                  ✅        │
└─────────────────────────────────────────────────────┘
```

### 7.2 Final Report Structure

**Executive Summary (1 page):**
- Test outcome: Success/Failure
- Primary metric result
- Recommendation: Launch/Don't launch
- Expected business impact

**Detailed Analysis (5-10 pages):**
1. Experiment setup
2. Sample characteristics
3. Statistical results (all metrics)
4. Subgroup analysis
5. Qualitative feedback
6. Technical performance
7. Risks and mitigations
8. Next steps

---

## 8. Risks & Mitigation

### 8.1 Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Model latency >2s | High | Pre-computation, caching |
| Model failures | High | Fallback to popular restaurants |
| Data pipeline issues | Medium | Monitoring, alerts |
| Incorrect test assignment | High | Validation checks, unit tests |

### 8.2 Business Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Restaurant complaints (unequal visibility) | Medium | Fair distribution monitoring |
| User backlash (privacy concerns) | High | Clear communication, opt-out |
| Revenue impact (lower AOV) | Medium | Track closely, set guardrails |
| Competitor copying | Low | Move fast, iterate |

### 8.3 Statistical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Selection bias | High | Randomization checks |
| Novelty effect | Medium | Extended test duration |
| Network effects | Low | User-level randomization |
| Simpson's paradox | Medium | Subgroup analysis |

---

## 9. Post-Launch Plan

### 9.1 Monitoring (First 4 Weeks)

**Week 1: Intensive Monitoring**
- Daily dashboard review
- Real-time alerts
- User feedback collection

**Week 2-4: Regular Monitoring**
- Weekly metric reviews
- Bi-weekly stakeholder updates
- Continuous model performance tracking

### 9.2 Iteration Plan

**Quick Wins (Month 1-2):**
- Tune model weights based on feedback
- Improve explanations clarity
- Fix edge cases discovered in production

**Medium-term (Month 3-6):**
- Add user feedback loop ("Not interested")
- Implement multi-armed bandit
- Real-time availability filtering

**Long-term (6+ months):**
- Deep learning models
- Multi-objective optimization
- Multi-city expansion

---

## 10. Stakeholder Communication

### 10.1 Pre-Launch Communication

**Audience:** Engineering, Product, Business, Marketing

**Message:**
- What: ML-powered restaurant recommendations
- Why: Reduce decision fatigue, improve discovery
- When: 2-week A/B test starting [date]
- Impact: Potentially 40% faster ordering

### 10.2 During Test

**Daily:** Email summary to core team  
**Weekly:** Presentation to leadership  
**Ad-hoc:** Slack updates on significant changes  

### 10.3 Post-Launch

**Success Announcement:**
- Internal all-hands presentation
- External blog post (if appropriate)
- Case study for portfolio

---

## Appendix A: SQL Queries

### Extract Primary Metric
```sql
-- Time to order for each user session
SELECT 
    user_id,
    test_group,
    session_id,
    TIMESTAMPDIFF(SECOND, 
        home_page_load_time, 
        order_placed_time
    ) / 60.0 AS time_to_order_minutes
FROM sessions
WHERE order_placed_time IS NOT NULL
    AND test_start_date <= order_placed_time
    AND order_placed_time <= test_end_date
    AND time_to_order_minutes < 30  -- Remove outliers
```

### Extract Secondary Metrics
```sql
-- Conversion rate
SELECT 
    test_group,
    COUNT(DISTINCT CASE WHEN order_placed THEN session_id END) / 
        COUNT(DISTINCT session_id) AS conversion_rate
FROM sessions
WHERE test_start_date <= session_start_time
GROUP BY test_group
```

---

**Document Owner:** Ayush Saxena  
**Review Date:** Pre-Launch  
**Status:** Approved for Testing