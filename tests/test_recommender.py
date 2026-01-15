"""
Unit tests for the Recommendation System
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path so imports work
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_generator import RestaurantDataGenerator
from feature_engineering import FeatureEngineer
from collaborative_filtering import CollaborativeFilteringRecommender
from content_based_filtering import ContentBasedRecommender
from hybrid_recommender import HybridRecommender

@pytest.fixture
def sample_data():
    """Create a tiny dataset for testing purposes"""
    # Create generator
    gen = RestaurantDataGenerator()

    # Create dummy users
    users = pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(10)],
        'total_orders': np.random.randint(1, 10, 10),
        'avg_order_value': np.random.uniform(100, 500, 10),
        'favorite_cuisine': ['North Indian'] * 10,
        'price_sensitivity': ['medium'] * 10,
        'avg_rating_given': [4.5] * 10,
        'dietary_preference': ['veg'] * 10,
        'preferred_meal_time': ['dinner'] * 10,
        'location_lat': [28.5] * 10,
        'location_lon': [77.1] * 10,
        'days_since_last_order': [5] * 10
    })
    
    restaurants = pd.DataFrame({
        'restaurant_id': [f'rest_{i}' for i in range(10)],
        'name': [f'Rest {i}' for i in range(10)],
        'cuisine_type': ['North Indian'] * 5 + ['Chinese'] * 5,
        'avg_rating': [4.2] * 10,
        'total_reviews': [100] * 10,
        'price_range': [2] * 10,
        'avg_delivery_time': [30] * 10,
        'is_veg_only': [True] * 10,
        'location_lat': [28.5] * 10,
        'location_lon': [77.1] * 10,
        'operating_hours': ['10AM-10PM'] * 10,
        'commission_rate': [0.2] * 10,
        'popularity_score': [0.8] * 10,
        'value_score': [1.0] * 10,
        'retention_rate': [0.5] * 10,
        'delivery_efficiency': [0.9] * 10
    })
    
    # Create dummy orders
    orders = pd.DataFrame({
        'order_id': range(20),
        'user_id': ['user_0', 'user_1'] * 10,
        'restaurant_id': ['rest_0', 'rest_1'] * 10,
        'order_value': [200.0] * 20,
        'order_timestamp': pd.to_datetime('2025-01-01'),
        'delivery_time': [30] * 20,
        'user_rating': [5.0] * 20,
        'cuisine_type': ['North Indian'] * 20
    })
    
    return users, restaurants, orders

def test_feature_engineering_pipeline(sample_data):
    """Test that feature engineer runs without errors"""
    users, restaurants, orders = sample_data
    engineer = FeatureEngineer(users, restaurants, orders)
    
    user_feat = engineer.create_user_features()
    assert not user_feat.empty
    assert 'price_preference_score' in user_feat.columns
    
    rest_feat = engineer.create_restaurant_features()
    assert not rest_feat.empty
    assert 'popularity_score' in rest_feat.columns
    
    interaction = engineer.create_user_restaurant_matrix()
    assert not interaction.empty
    assert interaction.shape == (2, 2) # based on dummy data users 0,1 and rest 0,1

def test_full_recommender_flow(sample_data):
    """Integration test for the Hybrid Recommender"""
    users, restaurants, orders = sample_data
    engineer = FeatureEngineer(users, restaurants, orders)
    
    # 1. Engineer features
    user_feat = engineer.create_user_features()
    rest_feat = engineer.create_restaurant_features()
    interaction = engineer.create_user_restaurant_matrix()
    
    # 2. Train CF Model
    cf = CollaborativeFilteringRecommender(interaction)
    cf.fit()
    assert cf.fitted is True
    
    # 3. Train CB Model
    cb = ContentBasedRecommender(rest_feat, user_feat)
    cb.fit()
    assert cb.fitted is True
    
    # 4. Hybrid Recommend
    hybrid = HybridRecommender(cf, cb, rest_feat, user_feat)
    
    # Get recs for user_0
    recs = hybrid.recommend('user_0', n_recommendations=5)
    
    # Should return dataframe with specific columns
    assert isinstance(recs, pd.DataFrame)
    assert 'restaurant_id' in recs.columns
    assert 'final_score' in recs.columns
    
    pass