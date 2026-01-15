# Edge Cases & Error Handling

**Project:** Restaurant Recommendation System  
**Author:** Ayush Saxena
**Date:** January 2026  
**Version:** 1.0

---

## Overview

This document catalogs edge cases, error scenarios, and handling strategies for the restaurant recommendation system. Proper edge case handling is critical for production reliability.

---

## 1. User-Related Edge Cases

### 1.1 New User (Cold Start)

**Scenario:** User has 0 order history

**Behavior:**
- Cannot use collaborative filtering (no interaction data)
- Must rely on content-based + contextual

**Handling:**
```python
if user_order_count == 0:
    # Trigger onboarding flow
    if user_completed_onboarding:
        # Use preferences from onboarding
        recommendations = cold_start_handler.onboarding_recommend(
            user_preferences,
            n=10
        )
    else:
        # Fallback to popular restaurants
        recommendations = cold_start_handler.popular_recommend(n=10)
```

**User Experience:**
- Show onboarding prompt (3 questions)
- Allow skip option (defaults to popular)
- Clear messaging: "Help us personalize your experience"

---

### 1.2 Inactive User (Stale Data)

**Scenario:** User hasn't ordered in 90+ days

**Issues:**
- Preferences may have changed
- Historical data less relevant
- Recency score approaches zero

**Handling:**
```python
if days_since_last_order > 90:
    # Reduce weight of historical preferences
    cf_weight = 0.2  # Down from 0.4
    cb_weight = 0.5  # Up from 0.35
    context_weight = 0.3  # Up from 0.25
    
    # Add "Welcome back!" messaging
    add_notification("We've updated recommendations based on what's popular now")
```

**Alternative Approach:**
- Treat as "semi-cold start"
- Blend old preferences with current trends

---

### 1.3 Power User (Extreme Order Count)

**Scenario:** User has 200+ orders

**Issues:**
- Interaction matrix dominated by this user
- May have ordered from 100+ restaurants
- Difficult to find "new" recommendations

**Handling:**
```python
if user_order_count > 100:
    # Increase novelty weight
    novelty_boost = 1.5
    
    # Expand "new restaurant" definition
    new_threshold_days = 180  # Haven't ordered in 6 months = "new"
    
    # Prioritize discovery over familiarity
    familiar_ratio = 0.2  # 20% familiar, 80% new
```

**User Experience:**
- Badge: "Foodie Explorer"
- Section: "Hidden Gems You Haven't Tried"
- Explicit novelty messaging

---

### 1.4 User with Extreme Preferences

**Scenario:** User only orders from 1 cuisine type (e.g., only North Indian)

**Issues:**
- All recommendations become same cuisine
- No diversity
- Filter bubble

**Handling:**
```python
if user_cuisine_concentration > 0.9:  # 90%+ from one cuisine
    # Force diversity
    max_per_cuisine = 2  # Down from 3
    
    # Add "You might also like" section
    recommendations += get_adjacent_cuisines(
        primary_cuisine,
        n=3
    )
    
    # Soften explanations
    # Instead of: "You love North Indian"
    # Use: "Based on your taste, you might enjoy..."
```

**Adjacent Cuisines Logic:**
```python
cuisine_similarity = {
    'North Indian': ['South Indian', 'Punjabi', 'Mughlai'],
    'Chinese': ['Thai', 'Japanese', 'Pan-Asian'],
    'Italian': ['Continental', 'Mediterranean', 'Pizza']
}
```

---

### 1.5 User with Conflicting Preferences

**Scenario:** User says "vegetarian" but orders from non-veg restaurants

**Issues:**
- Profile doesn't match behavior
- Which signal to trust?

**Handling:**
```python
if stated_preference != behavioral_preference:
    # Trust behavior over stated preference
    effective_preference = behavioral_preference
    
    # Suggest updating profile
    show_notification(
        "We noticed you often order from non-veg restaurants. "
        "Update your dietary preference for better recommendations?"
    )
```

**Priority Hierarchy:**
1. Recent behavior (last 10 orders)
2. Long-term behavior (last 6 months)
3. Stated preference (profile)

---

## 2. Restaurant-Related Edge Cases

### 2.1 New Restaurant (No Orders)

**Scenario:** Restaurant just joined platform, 0 orders

**Issues:**
- No collaborative signal
- No user ratings yet
- Cold start for restaurant

**Handling:**
```python
if restaurant_order_count == 0:
    # Use content-based features only
    score = content_based_score(restaurant, user)
    
    # Add "new restaurant" boost (exploration)
    if restaurant_days_on_platform < 30:
        score *= 1.2  # 20% boost
        add_badge("New on Platform")
```

**Gradual Integration:**
- Week 1: Show to 5% of compatible users
- Week 2-4: Increase to 20% if ratings positive
- Month 2+: Full integration if avg rating ≥3.5

---

### 2.2 Restaurant with Low Ratings

**Scenario:** Restaurant has avg rating <3.0

**Issues:**
- Poor user experience if recommended
- Guardrail metric: Maintain quality

**Handling:**
```python
# Hard filter: Never recommend below 3.0
if restaurant_avg_rating < 3.0:
    exclude_from_recommendations = True

# Soft filter: Penalize 3.0-3.5
if 3.0 <= restaurant_avg_rating < 3.5:
    score *= 0.7  # 30% penalty
```

**Exception:**
- If user has historically ordered from this restaurant (personal preference)
- Allow but add warning: "Rating has dropped since your last order"

---

### 2.3 Restaurant Temporarily Closed

**Scenario:** Restaurant is closed (maintenance, holidays, peak hours)

**Issues:**
- User frustration if recommended
- Wasted recommendation slot

**Handling:**
```python
# Real-time availability check (if API available)
if not restaurant.is_accepting_orders():
    exclude_from_recommendations = True

# Fallback: Use operating hours
current_time = datetime.now().time()
if not restaurant.is_open_at(current_time):
    exclude_from_recommendations = True
```

**Future Enhancement:**
- "Opens at 5 PM" badge for closed restaurants
- "Save for later" option

---

### 2.4 Restaurant Far from User

**Scenario:** Restaurant is 15+ km away

**Issues:**
- Long delivery time (>60 min)
- High delivery fees
- Cold food

**Handling:**
```python
if distance_km > 10:
    # Hard filter for portfolio version
    exclude_from_recommendations = True

# Production: Soft filter with warning
if 10 < distance_km < 15:
    score *= 0.5  # Heavy penalty
    add_warning("Longer delivery time expected")
```

**Context-Dependent:**
- Late night (1-3 AM): Relax distance constraint (few options)
- Peak hours: Tighten constraint (traffic delays)

---

### 2.5 Restaurant with Limited Menu

**Scenario:** Restaurant only has 3 menu items

**Issues:**
- May not satisfy all users
- Limited options for dietary restrictions

**Handling:**
```python
if restaurant_menu_item_count < 5:
    # Add badge: "Specialty Restaurant"
    # Lower weight slightly
    score *= 0.9
```

---

## 3. Model-Related Edge Cases

### 3.1 Model Prediction Failure

**Scenario:** Model throws exception or times out

**Issues:**
- User sees error or blank screen
- Lost revenue opportunity

**Handling:**
```python
try:
    recommendations = hybrid_model.recommend(
        user_id=user_id,
        n=10,
        timeout=2.0  # 2 second timeout
    )
except Exception as e:
    log_error(f"Model failure for user {user_id}: {e}")
    
    # Fallback: Popular restaurants
    recommendations = fallback_recommender.popular_restaurants(
        user_location=user_location,
        n=10
    )
    
    # Add subtle notice
    # "Showing popular restaurants in your area"
```

**Fallback Hierarchy:**
1. Cached recommendations (pre-computed)
2. Popular restaurants (by location)
3. Random sample (last resort)

---

### 3.2 All Recommendations Below Quality Threshold

**Scenario:** Model returns 10 restaurants, all with score <0.3

**Issues:**
- Low confidence predictions
- May not satisfy user

**Handling:**
```python
if max(recommendation_scores) < 0.3:
    # Blend with popular restaurants
    model_recs = recommendations[:5]  # Top 5 from model
    popular_recs = get_popular(n=5)   # Top 5 popular
    
    final_recs = interleave(model_recs, popular_recs)
```

---

### 3.3 No Recommendations After Filtering

**Scenario:** All model recommendations filtered out (distance, rating, etc.)

**Issues:**
- Empty recommendation list
- User sees "No restaurants found"

**Handling:**
```python
recommendations = apply_filters(candidate_restaurants)

if len(recommendations) == 0:
    # Relax filters progressively
    
    # Step 1: Relax distance (10km → 15km)
    recommendations = apply_filters(
        candidate_restaurants,
        max_distance=15
    )
    
    # Step 2: Relax rating (3.5 → 3.0)
    if len(recommendations) == 0:
        recommendations = apply_filters(
            candidate_restaurants,
            max_distance=15,
            min_rating=3.0
        )
    
    # Step 3: Show popular (no filters)
    if len(recommendations) == 0:
        recommendations = get_popular_restaurants(n=10)
        add_message("Showing popular restaurants (some may be far)")
```

---

### 3.4 Duplicate Recommendations

**Scenario:** Model returns same restaurant multiple times

**Issues:**
- Wasted recommendation slots
- Looks like a bug

**Handling:**
```python
# Deduplicate before returning
recommendations = list(dict.fromkeys(recommendations))  # Preserves order

# If <10 after deduplication, fill with next-best
if len(recommendations) < 10:
    additional = get_next_best_recommendations(
        exclude=recommendations,
        n=10 - len(recommendations)
    )
    recommendations.extend(additional)
```

---

## 4. Context-Related Edge Cases

### 4.1 Invalid Location Data

**Scenario:** User's GPS coordinates are (0, 0) or out of service area

**Issues:**
- Can't calculate distance
- Can't provide location-based recommendations

**Handling:**
```python
if not is_valid_location(user_lat, user_lon):
    # Use last known location
    if user_has_previous_location:
        user_location = get_last_known_location(user_id)
    else:
        # Use city center as default
        user_location = get_city_center(user_city)
    
    add_message("Using your city location. Enable GPS for better recommendations.")
```

---

### 4.2 Unusual Time (3 AM Order)

**Scenario:** User orders at 3 AM

**Issues:**
- Most restaurants closed
- Limited options
- Different food preferences (late-night cravings)

**Handling:**
```python
if 2 <= current_hour <= 5:  # 2 AM - 5 AM
    # Filter: Only show 24-hour restaurants
    restaurants = filter_24_hour(restaurants)
    
    # Boost late-night cuisine
    cuisine_boost = {
        'Fast Food': 1.5,
        'Street Food': 1.4,
        'Chinese': 1.3
    }
    
    # Relax quality threshold (fewer options)
    min_rating = 3.0  # Down from 3.5
```

---

### 4.3 Extreme Weather

**Scenario:** Heavy rain, storm warning, or extreme heat

**Issues:**
- Delivery delays
- Restaurant closures
- Safety concerns

**Handling:**
```python
if weather_alert_level == "severe":
    # Add prominent warning
    show_banner(
        "Weather Alert: Deliveries may be delayed or unavailable. "
        "Check with restaurant before ordering."
    )
    
    # Penalize outdoor/street food
    if restaurant_type in ['Street Food', 'Food Truck']:
        score *= 0.5
    
    # Boost reliable chains
    if restaurant_is_chain:
        score *= 1.2
```

---

### 4.4 Festival/Holiday

**Scenario:** Diwali, Holi, Christmas, or local festival

**Issues:**
- Many restaurants closed
- Different food preferences (festive meals)
- High demand, longer delivery times

**Handling:**
```python
if is_festival_day(current_date):
    # Boost festival-appropriate cuisines
    if festival == "Diwali":
        cuisine_boost = {
            'Desserts': 2.0,
            'North Indian': 1.5,
            'South Indian': 1.4
        }
    
    # Show "Open during festival" badge
    if restaurant_is_open:
        add_badge("Open Today")
    
    # Set expectations
    add_message("Festival rush: Deliveries may take longer than usual")
```

---

## 5. Data Quality Edge Cases

### 5.1 Missing User Features

**Scenario:** User profile incomplete (no dietary preference, no favorite cuisine)

**Handling:**
```python
# Use safe defaults
user_features = {
    'dietary_preference': user.dietary_preference or 'no_preference',
    'favorite_cuisine': user.favorite_cuisine or get_most_popular_cuisine(),
    'price_sensitivity': user.price_sensitivity or 'medium'
}
```

---

### 5.2 Inconsistent Restaurant Data

**Scenario:** Restaurant has rating=4.5 but 0 reviews

**Handling:**
```python
# Data validation on ingestion
if restaurant_avg_rating > 0 and restaurant_total_reviews == 0:
    log_warning(f"Inconsistent data for {restaurant_id}")
    
    # Don't use rating until reviews accumulate
    restaurant_avg_rating = None
    use_content_based_only = True
```

---

### 5.3 Extreme Outliers

**Scenario:** Order value = ₹50,000 (catering order) or delivery time = 300 min

**Handling:**
```python
# Remove outliers before model training
orders = orders[
    (orders['order_value'] >= 100) & 
    (orders['order_value'] <= 2000) &
    (orders['delivery_time'] >= 15) &
    (orders['delivery_time'] <= 90)
]
```

---

## 6. Business Logic Edge Cases

### 6.1 Restaurant with High Commission

**Scenario:** Platform wants to promote high-margin restaurants

**Conflict:** Business goal vs user satisfaction

**Handling (Ethical Approach):**
```python
# DO NOT boost based on commission
# Maintain user trust and long-term retention

# Optional: Paid placement section (clearly labeled)
# "Sponsored" section separate from recommendations
```

**Why This Matters:**
- User trust is paramount
- Short-term revenue gain < long-term reputation damage
- Recommendations should optimize for user, not business

---

### 6.2 Restaurant Inventory Constraint

**Scenario:** Restaurant only has 5 delivery slots left

**Issues:**
- If recommended to many users, won't fulfill
- User frustration

**Handling (Future Enhancement):**
```python
# Real-time inventory check
if restaurant_available_slots < 10:
    # Reduce recommendation frequency
    show_probability *= (restaurant_available_slots / 10)

# Fallback for MVP: Not implemented
# Requires real-time restaurant API integration
```

---

### 6.3 User in Banned/Fraud List

**Scenario:** User flagged for fraudulent behavior

**Handling:**
```python
if user_is_banned:
    # Don't provide service
    return error_response("Account suspended. Contact support.")

if user_is_flagged_for_fraud:
    # Provide service but log activity
    log_fraud_attempt(user_id, activity="recommendation_request")
    # Return recommendations (don't reveal detection)
    return recommendations
```

---

## 7. Concurrency & Race Conditions

### 7.1 User Orders While Browsing Recommendations

**Scenario:** User views recommendations, then orders from a different screen, then returns

**Issues:**
- Recommendations may now include just-ordered restaurant
- Seems like a mistake

**Handling:**
```python
# Check session state before displaying
recent_orders = get_orders_in_last_n_minutes(user_id, n=5)

# Exclude from recommendations
recommendations = [
    r for r in recommendations 
    if r.restaurant_id not in recent_orders
]
```

---

### 7.2 Multiple Concurrent Requests

**Scenario:** User rapidly refreshes page, triggering multiple recommendation calls

**Issues:**
- Wasted computation
- Potential race conditions

**Handling:**
```python
# Request deduplication
request_key = f"recs:{user_id}:{context_hash}"

# Check if request in progress
if redis.exists(f"processing:{request_key}"):
    # Return cached or wait
    return wait_for_completion(request_key)

# Set processing flag
redis.setex(f"processing:{request_key}", ttl=5, value=1)

try:
    recommendations = compute_recommendations()
    cache_recommendations(recommendations)
finally:
    redis.delete(f"processing:{request_key}")
```

---

## 8. Compliance & Privacy

### 8.1 User Requests Data Deletion (GDPR)

**Scenario:** User exercises "right to be forgotten"

**Handling:**
```python
def delete_user_data(user_id):
    # Delete personal data
    delete_from_db(f"users WHERE user_id = {user_id}")
    delete_from_db(f"orders WHERE user_id = {user_id}")
    
    # Anonymize in analytics
    anonymize_user_id(user_id)
    
    # Remove from recommendation models
    remove_from_interaction_matrix(user_id)
    
    # Flag for model re-training
    schedule_model_retrain()
```

---

### 8.2 Minor User (<18 Years Old)

**Scenario:** User is under 18

**Issues:**
- Different content policies
- Parental consent requirements

**Handling:**
```python
if user_age < 18:
    # Don't recommend alcohol-serving restaurants
    restaurants = exclude_alcohol_license(restaurants)
    
    # Require parental consent for account
    if not parental_consent_given:
        prompt_for_consent()
```

---

## 9. Error Handling Summary

### Error Hierarchy
```python
class RecommendationError(Exception):
    """Base exception for recommendation system"""
    pass

class UserNotFoundError(RecommendationError):
    """User doesn't exist in system"""
    fallback = "popular_restaurants"

class ModelTimeoutError(RecommendationError):
    """Model took too long to respond"""
    fallback = "cached_recommendations"

class NoRecommendationsError(RecommendationError):
    """No restaurants match criteria"""
    fallback = "relax_filters"

class DataQualityError(RecommendationError):
    """Data validation failed"""
    fallback = "skip_invalid_records"
```

### Centralized Error Handler
```python
def safe_recommend(user_id, context, n=10):
    """
    Fault-tolerant recommendation with automatic fallbacks
    """
    try:
        # Primary: ML model
        return hybrid_model.recommend(user_id, context, n)
    
    except ModelTimeoutError:
        # Fallback 1: Cached recommendations
        cached = get_cached_recommendations(user_id)
        if cached:
            return apply_context(cached, context)[:n]
    
    except (UserNotFoundError, NoRecommendationsError):
        # Fallback 2: Popular restaurants
        return popular_recommender(user_location, n)
    
    except Exception as e:
        # Fallback 3: Random (last resort)
        log_critical_error(f"All fallbacks failed: {e}")
        return random_restaurants(n)
    
    finally:
        # Always log
        log_recommendation_attempt(user_id, success=True)
```

---

## 10. Testing Edge Cases

### Unit Tests
```python
def test_cold_start_user():
    user = create_user(order_count=0)
    recs = recommender.recommend(user.id)
    assert len(recs) == 10
    assert all(r.rating >= 3.5 for r in recs)

def test_invalid_location():
    user = create_user(location=(0, 0))
    recs = recommender.recommend(user.id)
    assert len(recs) == 10  # Should not crash

def test_model_timeout():
    with mock.patch('model.predict', side_effect=TimeoutError):
        recs = recommender.recommend(user.id)
        assert len(recs) == 10  # Fallback should work
```

---

## Summary: Edge Case Priorities

### P0 (Must Handle)
✅ New user (cold start)  
✅ Model failure/timeout  
✅ Invalid location  
✅ No recommendations after filtering  

### P1 (Should Handle)
✅ Inactive user (stale data)  
✅ Restaurant temporarily closed  
✅ Extreme weather  
✅ Data quality issues  

### P2 (Nice to Have)
⚠️ Power user (extreme orders)  
⚠️ Unusual time (3 AM)  
⚠️ Festival/holiday  
⚠️ Real-time inventory  

---

**Document Owner:** Ayush Saxena 
**Last Updated:** January 2026  
**Review Cycle:** Quarterly or post-incident