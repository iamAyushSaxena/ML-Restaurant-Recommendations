"""
Configuration file for Restaurant Recommendation System
Author: Ayush Saxena
Date: January 2026
"""

import os
from pathlib import Path
from datetime import datetime, timedelta

# ===== PROJECT PATHS =====
PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
DASHBOARDS_DIR = OUTPUT_DIR / "dashboards"

# Create directories
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SYNTHETIC_DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    DASHBOARDS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ===== BUSINESS CONTEXT =====
# Focused scope: Home feed recommendations for repeat users
TARGET_MARKET = "Urban India (Tier 1 cities)"
USER_BASE = 50000  # Synthetic user base
RESTAURANT_BASE = 500  # Restaurants in dataset
HISTORICAL_ORDERS = 200000  # Total orders for training

# ===== PROBLEM FRAMING =====
PROBLEM_STATEMENT = """
Food delivery users spend excessive time browsing (8-12 minutes) due to 
overwhelming restaurant choices and lack of contextual prioritization. 
This increases decision fatigue, cart abandonment (~30%), and repeat ordering 
from a small set of familiar restaurants (low discovery).
"""

PRODUCT_GOAL = """
Reduce time-to-order by 40% (10 min → 6 min) while increasing restaurant 
discovery (2+ new restaurants/user/month) and improving conversion by 15%.
"""

# ===== SUCCESS METRICS =====
# Primary Metric (Hero Metric)
PRIMARY_METRIC = {
    "name": "Time to Order",
    "baseline": 600,  # 10 minutes in seconds
    "target": 360,  # 6 minutes in seconds
    "improvement": -40,  # -40% reduction
    "measurement": "Time from home page load to order placement",
}

# Secondary Metrics
SECONDARY_METRICS = {
    "order_conversion_rate": {
        "baseline": 0.12,  # 12%
        "target": 0.138,  # 13.8% (15% relative improvement)
        "measurement": "Orders placed / Sessions with recommendations",
    },
    "restaurant_discovery_rate": {
        "baseline": 1.2,  # New restaurants per user per month
        "target": 2.0,
        "measurement": "Unique new restaurants ordered from per user",
    },
    "repeat_order_rate": {
        "baseline": 0.65,  # 65% repeat from same restaurants
        "target": 0.55,  # 55% (more discovery)
        "measurement": "Orders from previously ordered restaurants",
    },
}

# Guardrail Metrics (Must not degrade)
GUARDRAIL_METRICS = {
    "avg_delivery_time": {
        "baseline": 35,  # minutes
        "threshold": 38,  # Don't go above this
        "measurement": "Average delivery time for recommended restaurants",
    },
    "order_cancellation_rate": {
        "baseline": 0.05,  # 5%
        "threshold": 0.06,
        "measurement": "Cancellations / Total orders",
    },
    "user_dissatisfaction": {
        "baseline": 0.08,  # 8% negative feedback
        "threshold": 0.10,
        "measurement": "Negative ratings on recommended restaurants",
    },
}

# ===== RECOMMENDATION PARAMETERS =====
# How many recommendations to show
N_RECOMMENDATIONS_HOME_FEED = 10
N_RECOMMENDATIONS_PERSONALIZED = 20

# Model weights (for hybrid system)
MODEL_WEIGHTS = {
    "collaborative_filtering": 0.40,  # 40% weight
    "content_based": 0.35,  # 35% weight
    "contextual": 0.25,  # 25% weight (time, weather, etc.)
}

# Minimum orders required for collaborative filtering
MIN_ORDERS_FOR_CF = 3  # Users with < 3 orders use content-based only

# ===== USER FEATURES =====
USER_FEATURE_COLUMNS = [
    "user_id",
    "total_orders",
    "avg_order_value",
    "favorite_cuisine",
    "price_sensitivity",  # low, medium, high
    "avg_rating_given",
    "dietary_preference",  # veg, non_veg, vegan, no_preference
    "preferred_meal_time",  # breakfast, lunch, dinner, snacks
    "location_lat",
    "location_lon",
    "days_since_last_order",
]

# ===== RESTAURANT FEATURES =====
RESTAURANT_FEATURE_COLUMNS = [
    "restaurant_id",
    "name",
    "cuisine_type",
    "avg_rating",
    "total_reviews",
    "price_range",  # 1-4 (₹ to ₹₹₹₹)
    "avg_delivery_time",
    "is_veg_only",
    "location_lat",
    "location_lon",
    "popular_dishes",
    "operating_hours",
    "commission_rate",  # Platform's cut
]

# ===== CUISINE TYPES =====
CUISINE_TYPES = [
    "North Indian",
    "South Indian",
    "Chinese",
    "Italian",
    "Continental",
    "Fast Food",
    "Street Food",
    "Biryani",
    "Desserts",
    "Beverages",
    "Healthy",
    "Cafe",
]

# ===== CONTEXTUAL FACTORS =====
CONTEXT_FEATURES = {
    "time_of_day": ["breakfast", "lunch", "evening_snack", "dinner", "late_night"],
    "day_of_week": ["weekday", "weekend"],
    "weather": ["clear", "rainy", "hot"],  # Simplified
    "occasion": ["regular", "weekend_treat", "party_order"],
}

# ===== EXPLAINABILITY TEMPLATES =====
EXPLANATION_TEMPLATES = {
    "user_history": "You've ordered {cuisine} {count} times",
    "popular_choice": "Popular among users who like {cuisine}",
    "highly_rated": "Rated {rating}/5 by {reviews} customers",
    "new_discovery": "New restaurant that matches your taste",
    "contextual": "Perfect for {meal_time}",
    "distance": "Only {distance} km away, delivers in {time} min",
    "trending": "Trending in your area this week",
}

# ===== COLD START STRATEGIES =====
COLD_START_CONFIG = {
    "onboarding_questions": [
        {
            "question": "What type of food do you prefer?",
            "options": ["Veg", "Non-Veg", "Vegan", "No Preference"],
            "weight": 0.4,
        },
        {
            "question": "Which cuisines do you enjoy? (Select up to 3)",
            "options": CUISINE_TYPES,
            "multi_select": True,
            "weight": 0.35,
        },
        {
            "question": "What's your typical budget per meal?",
            "options": ["₹0-200", "₹200-400", "₹400-600", "₹600+"],
            "weight": 0.25,
        },
    ],
    "fallback_strategy": "popular_restaurants",  # If no preferences
    "min_interaction_threshold": 1,  # Orders needed before personalization
}

# ===== MODEL EVALUATION METRICS =====
MODEL_METRICS = {
    "precision_at_k": [1, 5, 10],  # Precision at top-1, top-5, top-10
    "recall_at_k": [5, 10, 20],
    "ndcg_at_k": [5, 10],  # Normalized Discounted Cumulative Gain
    "hit_rate": True,  # Did user order from top-N recommendations?
    "diversity": True,  # Cuisine diversity in recommendations
    "novelty": True,  # How many new restaurants recommended
}

# ===== A/B TEST CONFIGURATION =====
AB_TEST_CONFIG = {
    "test_duration_days": 14,
    "control_group_size": 0.5,  # 50-50 split
    "treatment_group_size": 0.5,
    "min_sample_size_per_group": 10000,
    "significance_level": 0.05,
    "statistical_power": 0.8,
}

# ===== VISUALIZATION SETTINGS =====
COLOR_SCHEME = {
    "primary": "#E23744",  # Zomato red
    "secondary": "#FFFFFF",  # White
    "accent": "#CB202D",  # Dark red
    "background": "#FAFAFA",  # Light gray
    "text": "#1C1C1C",  # Dark gray
    "success": "#48C479",  # Green
    "warning": "#FFC043",  # Orange
    "info": "#3B5998",  # Blue
}

# Chart settings
CHART_STYLE = {
    "font_family": "Roboto, sans-serif",
    "title_size": 16,
    "label_size": 12,
    "dpi": 300,
}

# ===== OUT OF SCOPE (V1) =====
OUT_OF_SCOPE = [
    "Dynamic pricing optimization",
    "Restaurant commission strategies",
    "Courier assignment logic",
    "Long-term personalization (cross-month patterns)",
    "Promotional campaigns",
    "Multi-city rollout strategy",
    "Restaurant onboarding flow",
]

print("✅ Configuration loaded successfully")
print(f"📁 Project Root: {PROJECT_ROOT}")
print(f"🎯 Primary Metric: {PRIMARY_METRIC['name']}")
print(f"📊 Target: {PRIMARY_METRIC['improvement']}% improvement")
print(f"🍽️  Restaurants: {RESTAURANT_BASE}")
print(f"👥 Users: {USER_BASE}")
print(f"📦 Historical Orders: {HISTORICAL_ORDERS}")
