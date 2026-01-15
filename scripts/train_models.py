"""
Master script to train all models
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_generator import RestaurantDataGenerator
from feature_engineering import FeatureEngineer
from collaborative_filtering import CollaborativeFilteringRecommender
from content_based_filtering import ContentBasedRecommender
from hybrid_recommender import HybridRecommender
import pandas as pd

def main():
    print("="*80)
    print(" RESTAURANT RECOMMENDATION SYSTEM - MODEL TRAINING")
    print("="*80)
    print()
    
    # Step 1: Generate synthetic data
    print("STEP 1: Generating Synthetic Data")
    print("-" * 80)
    generator = RestaurantDataGenerator()
    generator.generate_restaurants()
    generator.generate_users()
    generator.generate_orders()
    generator.save_data()
    generator.generate_summary_stats()
    
    print("\n" + "="*80)
    
    # Step 2: Feature engineering
    print("\nSTEP 2: Feature Engineering")
    print("-" * 80)
    
    users_df = pd.read_csv('data/synthetic/users.csv')
    restaurants_df = pd.read_csv('data/synthetic/restaurants.csv')
    orders_df = pd.read_csv('data/synthetic/orders.csv')
    orders_df['order_timestamp'] = pd.to_datetime(orders_df['order_timestamp'])
    
    engineer = FeatureEngineer(users_df, restaurants_df, orders_df)
    user_features = engineer.create_user_features()
    restaurant_features = engineer.create_restaurant_features()
    interaction_matrix = engineer.create_user_restaurant_matrix()
    engineer.save_features()
    
    print("\n" + "="*80)
    
    # Step 3: Train collaborative filtering
    print("\nSTEP 3: Training Collaborative Filtering Model")
    print("-" * 80)
    
    cf_model = CollaborativeFilteringRecommender(interaction_matrix)
    cf_model.fit()
    cf_model.save_model()
    
    print("\n" + "="*80)
    
    # Step 4: Train content-based filtering
    print("\nSTEP 4: Training Content-Based Filtering Model")
    print("-" * 80)
    
    cb_model = ContentBasedRecommender(restaurant_features, user_features)
    cb_model.fit()
    cb_model.save_model()
    
    print("\n" + "="*80)
    
    # Step 5: Create hybrid model
    print("\nSTEP 5: Creating Hybrid Model")
    print("-" * 80)
    
    hybrid_model = HybridRecommender(cf_model, cb_model, restaurant_features, user_features)
    hybrid_model.save_model()
    
    print("\n" + "="*80)
    print("\n✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print("\nYou can now run the Streamlit app with:")
    print("   streamlit run app/streamlit_app.py")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()