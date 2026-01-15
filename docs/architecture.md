# System Architecture: Restaurant Recommendation System

**Version:** 1.0  
**Last Updated:** January 2026  
**Author:** Ayush Saxena

---

## Overview

This document describes the technical architecture of the ML-powered restaurant recommendation system, including data flow, model components, and system design decisions.

---

## High-Level Architecture
```
┌───────────────────────────────────────────────────────────┐
│                           USER                            │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│                        API LAYER                          │
│  • Request validation                                     │
│  • User authentication                                    │
│  • Context extraction (time, location, weather)           │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│                  RECOMMENDATION ENGINE                    │
│                                                           │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  │
│  │ Collaborative │  │ Content-Based │  │  Contextual   │  │
│  │  Filtering    │  │   Filtering   │  │    Scoring    │  │
│  │   (40%)       │  │    (35%)      │  │    (25%)      │  │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘  │
│          │                  │                  │          │
│          └──────────────────┴──────────────────┘          │
│                             │                             │
│                             ▼                             │
│                     ┌──────────────┐                      │
│                     │    Hybrid    │                      │
│                     │    Ranker    │                      │
│                     └───────┬──────┘                      │
│                             │                             │
│                             ▼                             │
│                   ┌─────────────────┐                     │
│                   │   Diversity &   │                     │
│                   │   Filter Layer  │                     │
│                   └────────┬────────┘                     │
└────────────────────────────│──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│                  EXPLAINABILITY ENGINE                    │
│  • Generate explanation for each recommendation           │
│  • Combine signals from all models                        │
└────────────────────┬──────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────────────────┐
│                      RESPONSE                             │
│  • Top-N recommendations                                  │
│  • Explanations                                           │
│  • Metadata (rating, delivery time, etc.)                 │
└───────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Data Layer

#### User Features
```python
{
    'user_id': str,
    'total_orders': int,
    'avg_order_value': float,
    'favorite_cuisine': str,
    'dietary_preference': str,  # veg, non_veg, vegan, no_preference
    'price_sensitivity': str,    # low, medium, high
    'location_lat': float,
    'location_lon': float,
    'order_history': List[str]  # Restaurant IDs
}
```

#### Restaurant Features
```python
{
    'restaurant_id': str,
    'name': str,
    'cuisine_type': str,
    'avg_rating': float,
    'total_reviews': int,
    'price_range': int,  # 1-4
    'avg_delivery_time': int,
    'is_veg_only': bool,
    'location_lat': float,
    'location_lon': float,
    'popularity_score': float
}
```

#### Interaction Matrix
```python
# Sparse matrix: Users × Restaurants
# Values: Weighted interaction scores
interaction_matrix = pd.DataFrame(
    index=user_ids,
    columns=restaurant_ids,
    values=interaction_scores
)

# Interaction score formula:
score = (
    0.4 × log1p(order_count) +
    0.3 × (avg_rating / 5.0) +
    0.3 × exp(-days_since_last / 30)
)
```

---

### 2. Collaborative Filtering Module

**Algorithm:** User-based collaborative filtering with cosine similarity

**Process:**
1. Compute user-user similarity matrix (50K × 50K)
2. For target user, find top-30 most similar users
3. Aggregate restaurant scores from similar users (weighted by similarity)
4. Normalize scores to [0, 1] range

**Implementation:**
```python
class CollaborativeFilteringRecommender:
    def fit(self):
        # Compute cosine similarity between all users
        self.user_similarity = cosine_similarity(interaction_matrix)
    
    def recommend(self, user_id, n=10):
        # Get similar users
        similar_users = self.get_similar_users(user_id, k=30)
        
        # Aggregate scores
        restaurant_scores = {}
        for similar_user, similarity in similar_users:
            for restaurant, score in user_scores[similar_user]:
                restaurant_scores[restaurant] += score * similarity
        
        # Normalize and return top-N
        return sorted(restaurant_scores, key=lambda x: x[1], reverse=True)[:n]
```

**Strengths:**
- Discovers unexpected matches (e.g., Italian + Thai → Japanese)
- Handles complex preference patterns
- Improves with more data

**Weaknesses:**
- Cold start: Fails for users with <3 orders
- Computational cost: O(n²) similarity computation
- Sparsity: 99%+ of user-restaurant pairs empty

**Mitigation:**
- Pre-compute similarity matrix (offline)
- Use sparse matrix representation
- Fallback to content-based for cold start

---

### 3. Content-Based Filtering Module

**Algorithm:** Feature matching with weighted scoring

**Process:**
1. Create restaurant feature vectors (8 dimensions)
2. Match against user preference profile
3. Calculate weighted similarity score

**Feature Weights:**
```python
score = (
    0.40 × cuisine_match_score +
    0.25 × price_match_score +
    0.20 × rating_normalized +
    0.15 × delivery_efficiency
)
```

**Why These Weights?**
- **40% Cuisine:** Strongest preference signal
- **25% Price:** Budget is hard constraint
- **20% Rating:** Quality threshold
- **15% Delivery:** Convenience factor

**Implementation:**
```python
class ContentBasedRecommender:
    def recommend(self, user_profile, n=10):
        scores = []
        for restaurant in candidates:
            score = self._calculate_match_score(user_profile, restaurant)
            scores.append((restaurant, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:n]
    
    def _calculate_match_score(self, user, restaurant):
        # Cuisine match
        cuisine_score = 1.0 if restaurant.cuisine == user.favorite_cuisine else 0.0
        
        # Price match
        price_diff = abs(restaurant.price_range - user.price_preference)
        price_score = max(0, 1 - price_diff / 3)
        
        # Rating
        rating_score = restaurant.avg_rating / 5.0
        
        # Delivery
        delivery_score = 1 - (restaurant.delivery_time - 20) / 40
        
        return 0.40*cuisine_score + 0.25*price_score + 0.20*rating_score + 0.15*delivery_score
```

**Strengths:**
- Works for new users (no history needed)
- Explainable (clear feature matching)
- Fast inference (no user similarity needed)

**Weaknesses:**
- No discovery (only matches known preferences)
- Misses collaborative signals
- Over-specialization risk (filter bubble)

---

### 4. Contextual Scoring Module

**Factors Considered:**
1. Time of day
2. Weather conditions
3. Day of week
4. Distance from user

**Time-Based Boosting:**
```python
time_boost_map = {
    'breakfast': {
        'South Indian': 1.3,
        'Cafe': 1.4,
        'Beverages': 1.3
    },
    'lunch': {
        'North Indian': 1.2,
        'Biryani': 1.3,
        'Chinese': 1.1
    },
    'dinner': {
        'North Indian': 1.3,
        'Biryani': 1.4,
        'Chinese': 1.2,
        'Continental': 1.1
    },
    'late_night': {
        'Fast Food': 1.5,
        'Street Food': 1.4,
        'Chinese': 1.2
    }
}
```

**Weather-Based Boosting:**
```python
weather_boost_map = {
    'rainy': {
        'Street Food': 0.6,   # Penalize (outdoor)
        'Fast Food': 1.3,     # Boost (comfort food)
        'Chinese': 1.2
    },
    'hot': {
        'Beverages': 1.5,
        'Desserts': 1.3,
        'South Indian': 1.1   # Lighter food
    }
}
```

**Distance Penalty:**
```python
# Exponential decay: closer is much better
distance_score = exp(-distance_km / 3.0)

# 3km half-life:
# 0 km: score = 1.0
# 3 km: score = 0.37
# 6 km: score = 0.14
# 10 km: score = 0.04
```

---

### 5. Hybrid Ranker

**Weight Combination:**
```python
# For users with ≥3 orders (CF eligible)
final_score = (
    0.40 × cf_score +
    0.35 × cb_score +
    0.25 × contextual_score
)

# For users with <3 orders (cold start)
final_score = (
    0.75 × cb_score +
    0.25 × contextual_score
)
```

**Ranking Algorithm:**
```python
def hybrid_recommend(user_id, context, n=10):
    # Step 1: Get candidate scores from each model
    cf_scores = cf_model.recommend(user_id, n=50)
    cb_scores = cb_model.recommend(user_id, n=50)
    contextual_scores = context_model.score(candidates, context)
    
    # Step 2: Merge scores
    all_candidates = merge_candidates(cf_scores, cb_scores)
    
    # Step 3: Apply contextual boost
    for candidate in all_candidates:
        candidate.score += contextual_scores[candidate.id]
    
    # Step 4: Compute final hybrid score
    for candidate in all_candidates:
        candidate.final_score = (
            weights['cf'] * candidate.cf_score +
            weights['cb'] * candidate.cb_score +
            weights['context'] * candidate.contextual_score
        )
    
    # Step 5: Filter
    candidates = apply_filters(all_candidates)
    
    # Step 6: Ensure diversity
    candidates = ensure_diversity(candidates, max_per_cuisine=3)
    
    # Step 7: Rank and return top-N
    return sorted(candidates, key=lambda x: x.final_score, reverse=True)[:n]
```

---

### 6. Filtering & Business Rules

**Hard Filters (Applied Always):**
- Minimum rating: ≥3.0
- Maximum distance: ≤10 km
- Restaurant status: Open and accepting orders
- Dietary restrictions: If user is veg/vegan

**Soft Rules (Applied for Diversity):**
- Maximum 3 restaurants per cuisine in top-10
- If user ordered from 5+ cuisines, show at least 4 different cuisines
- Balance familiar (40%) vs new (60%) restaurants

**Implementation:**
```python
def apply_filters(candidates):
    # Hard filters
    filtered = [
        c for c in candidates
        if c.rating >= 3.0 and
           c.distance <= 10 and
           c.is_open
    ]
    
    # Dietary filter (if applicable)
    if user.dietary_preference == 'veg':
        filtered = [c for c in filtered if c.is_veg_only]
    
    return filtered

def ensure_diversity(candidates, max_per_cuisine=3):
    diverse_results = []
    cuisine_count = {}
    
    for candidate in sorted(candidates, key=lambda x: x.score, reverse=True):
        cuisine = candidate.cuisine_type
        if cuisine_count.get(cuisine, 0) < max_per_cuisine:
            diverse_results.append(candidate)
            cuisine_count[cuisine] = cuisine_count.get(cuisine, 0) + 1
        
        if len(diverse_results) >= 10:
            break
    
    return diverse_results
```

---

### 7. Explainability Engine

**Explanation Generation Process:**
```python
def generate_explanation(user_id, restaurant_id, context):
    reasons = []
    
    # 1. User history match
    if restaurant.cuisine in user.order_history_cuisines:
        reasons.append({
            'type': 'user_history',
            'weight': 'high',
            'text': f"You've ordered {cuisine} {count} times"
        })
    
    # 2. Collaborative signal
    if restaurant in similar_users_favorites:
        reasons.append({
            'type': 'collaborative',
            'weight': 'high',
            'text': "Popular among users with similar taste"
        })
    
    # 3. Quality signal
    if restaurant.rating >= 4.0:
        reasons.append({
            'type': 'quality',
            'weight': 'medium',
            'text': f"Rated {rating}/5 by {reviews} customers"
        })
    
    # 4. Contextual relevance
    if is_contextually_relevant(restaurant, context):
        reasons.append({
            'type': 'contextual',
            'weight': 'medium',
            'text': f"Perfect for {context.time_of_day}"
        })
    
    # 5. Proximity
    if restaurant.delivery_time <= 30:
        reasons.append({
            'type': 'proximity',
            'weight': 'low',
            'text': f"Quick delivery in ~{time} minutes"
        })
    
    # Sort by weight and create final explanation
    reasons = sorted(reasons, key=lambda x: weight_order[x['weight']])
    
    return {
        'primary_reason': reasons[0]['text'],
        'supporting_reasons': [r['text'] for r in reasons[1:3]],
        'full_text': format_explanation(reasons)
    }
```

---

## Data Flow Diagram
```
User Request
    │
    ├─ Extract Context
    │  ├─ Time of day (from request timestamp)
    │  ├─ User location (from GPS or saved address)
    │  ├─ Weather (from external API)
    │  └─ Day of week
    │
    ├─ Load User Data
    │  ├─ User profile (preferences, dietary)
    │  ├─ Order history (last 6 months)
    │  └─ Interaction scores
    │
    ├─ Check Eligibility
    │  ├─ If orders ≥ 3: Use full hybrid
    │  └─ If orders < 3: Use cold start
    │
    ├─ Generate Scores
    │  ├─ CF Model → cf_scores
    │  ├─ CB Model → cb_scores
    │  └─ Context Module → contextual_scores
    │
    ├─ Combine Scores
    │  └─ final_score = weighted_sum(cf, cb, context)
    │
    ├─ Apply Filters
    │  ├─ Rating ≥ 3.0
    │  ├─ Distance ≤ 10 km
    │  └─ Dietary restrictions
    │
    ├─ Ensure Diversity
    │  └─ Max 3 per cuisine
    │
    ├─ Generate Explanations
    │  └─ For each top-N recommendation
    │
    └─ Return Response
       ├─ Restaurant list (name, rating, cuisine, etc.)
       ├─ Explanations
       └─ Metadata
```

---

## Scalability Considerations

### Current Scale (Portfolio Project)
- 50K users
- 500 restaurants
- 200K orders
- Response time: <2 seconds

### Production Scale Estimates

**Expected Load:**
- 5M active users
- 50K restaurants
- 100M orders/month
- 10K requests/second (peak)

**Bottlenecks & Solutions:**

1. **User Similarity Computation (O(n²))**
   - **Problem:** 5M × 5M matrix = 25 trillion comparisons
   - **Solution:** Approximate nearest neighbors (Annoy, FAISS)
   - **Improvement:** O(n²) → O(n log n)

2. **Real-Time Inference**
   - **Problem:** Can't compute recommendations on each request
   - **Solution:** Pre-compute top-100 for each user, refresh daily
   - **Hybrid:** Real-time contextual boost on pre-computed list

3. **Cold Start at Scale**
   - **Problem:** 30% new users/month = 1.5M cold starts
   - **Solution:** Vectorized content-based scoring
   - **Caching:** Popular restaurant lists by city/cuisine

4. **Model Update Frequency**
   - **Current:** Weekly batch re-training
   - **Production:** Daily batch + online learning
   - **Approach:** Incremental updates, not full re-train

---

## Technology Stack

### Current Implementation (Portfolio)
- **Language:** Python 3.8+
- **ML Libraries:** scikit-learn, pandas, numpy, scipy
- **Web Framework:** Streamlit
- **Data Storage:** CSV files (synthetic data)

### Production Stack (Recommended)

**Data Layer:**
- **Database:** PostgreSQL (user/restaurant data), Redis (caching)
- **Data Warehouse:** BigQuery or Snowflake (order history)
- **Feature Store:** Feast or Tecton

**ML Layer:**
- **Training:** Python, scikit-learn, XGBoost
- **Serving:** TensorFlow Serving or custom Flask API
- **Experimentation:** MLflow or Weights & Biases

**Infrastructure:**
- **Compute:** Kubernetes (auto-scaling)
- **Storage:** S3 or GCS (model artifacts)
- **Monitoring:** Prometheus + Grafana

---

## Monitoring & Observability

### Model Performance Metrics
- Precision@10, Recall@10, NDCG@10
- Hit Rate (daily, weekly)
- Diversity Score
- Latency (p50, p95, p99)

### Business Metrics
- Time-to-order (primary)
- Conversion rate
- Discovery rate
- User satisfaction (ratings, surveys)

### Operational Metrics
- API response time
- Error rate
- Model prediction success rate
- Cache hit rate

### Alerts
- Latency >2 seconds (95th percentile)
- Error rate >1%
- Hit Rate drops >10% (week-over-week)
- Model prediction failures >5%

---

## Security & Privacy

### Data Privacy
- PII encryption at rest and in transit
- User consent for preference collection
- Data retention: 6 months of order history

### Model Privacy
- No restaurant-specific promotion bias
- Fair visibility distribution
- Regular bias audits (cuisine, price, location)

---

## Future Architecture Enhancements

### Phase 2 (V2)
- Real-time availability filtering
- Multi-armed bandit for exploration
- User feedback loop ("Not interested")
- A/B testing framework

### Phase 3 (V3)
- Deep learning embeddings (BERT for restaurants)
- Graph neural networks (user-restaurant-cuisine)
- Reinforcement learning (long-term user satisfaction)
- Multi-objective optimization

---

**Document Owner:** Ayush Saxena 
**Last Updated:** January 2026