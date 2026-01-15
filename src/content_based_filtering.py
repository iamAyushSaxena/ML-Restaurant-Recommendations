"""
Content-Based Filtering Recommender
Recommends restaurants based on restaurant attributes and user preferences
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pickle
from typing import Tuple, Dict, List
from config import *


class ContentBasedRecommender:
    """
    Content-based filtering using restaurant features
    """
    
    def __init__(self, restaurant_features: pd.DataFrame, user_features: pd.DataFrame):
        """
        Args:
            restaurant_features: Restaurant attributes
            user_features: User preferences
        """
        self.restaurant_features = restaurant_features
        self.user_features = user_features
        self.restaurant_vectors = None
        self.scaler = StandardScaler()
        self.fitted = False
        
    def fit(self):
        """
        Create restaurant feature vectors
        """
        print("🔧 Creating restaurant feature vectors...")
        
        # Select numerical features for similarity computation
        feature_cols = [
            'avg_rating',
            'price_range',
            'avg_delivery_time',
            'is_veg_only_int',
            'cuisine_type_encoded',
            'popularity_score',
            'value_score',
            'delivery_efficiency'
        ]
        
        # Create feature matrix
        restaurant_matrix = self.restaurant_features[feature_cols].copy()
        
        # Fill any missing values
        restaurant_matrix = restaurant_matrix.fillna(0)
        
        # Standardize features
        restaurant_matrix_scaled = self.scaler.fit_transform(restaurant_matrix)
        
        # Store as DataFrame for easier indexing
        self.restaurant_vectors = pd.DataFrame(
            restaurant_matrix_scaled,
            index=self.restaurant_features['restaurant_id'],
            columns=feature_cols
        )
        
        self.fitted = True
        print(f"✅ Restaurant vectors created: {self.restaurant_vectors.shape}")
    
    def find_similar_restaurants(self, restaurant_id: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Find restaurants similar to a given restaurant
        
        Args:
            restaurant_id: Target restaurant ID
            k: Number of similar restaurants to return
            
        Returns:
            List of (restaurant_id, similarity_score) tuples
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if restaurant_id not in self.restaurant_vectors.index:
            return []
        
        # Get target restaurant vector
        target_vector = self.restaurant_vectors.loc[restaurant_id].values.reshape(1, -1)
        
        # Compute similarity with all restaurants
        similarities = cosine_similarity(target_vector, self.restaurant_vectors.values)[0]
        
        # Create DataFrame for easier sorting
        similarity_df = pd.DataFrame({
            'restaurant_id': self.restaurant_vectors.index,
            'similarity': similarities
        })
        
        # Remove the target restaurant itself
        similarity_df = similarity_df[similarity_df['restaurant_id'] != restaurant_id]
        
        # Sort and return top-k
        top_k = similarity_df.sort_values('similarity', ascending=False).head(k)
        return [(rid, sim) for rid, sim in zip(top_k['restaurant_id'], top_k['similarity'])]
    
    def recommend(self, user_id: str, n_recommendations: int = 10,
                  user_order_history: List[str] = None) -> pd.DataFrame:
        """
        Generate recommendations based on user's profile and order history
        
        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations to return
            user_order_history: List of restaurant IDs user has ordered from
            
        Returns:
            DataFrame with restaurant_id and content_score
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get user preferences
        if user_id not in self.user_features['user_id'].values:
            # Cold start: use popularity-based recommendations
            return self._cold_start_recommend(n_recommendations)
        
        user_profile = self.user_features[self.user_features['user_id'] == user_id].iloc[0]
        
        # Start with all restaurants
        candidates = self.restaurant_features.copy()
        
        # Apply hard filters based on user preferences
        
        # 1. Dietary filter
        if user_profile['dietary_preference'] in ['veg', 'vegan']:
            candidates = candidates[candidates['is_veg_only'] == True]
        
        # If too few candidates after filtering, relax constraints
        if len(candidates) < 20:
            candidates = self.restaurant_features.copy()
        
        # Calculate content-based scores
        scores = []
        
        for _, restaurant in candidates.iterrows():
            score = 0.0
            
            # 1. Cuisine match (40% weight)
            if restaurant['cuisine_type'] == user_profile['favorite_cuisine']:
                score += 0.40
            elif restaurant['cuisine_type'] == user_profile.get('most_ordered_cuisine'):
                score += 0.30
            
            # 2. Price match (25% weight)
            price_diff = abs(restaurant['price_range'] - user_profile['price_preference_score'])
            price_score = max(0, 1 - (price_diff / 3))  # Normalize
            score += 0.25 * price_score
            
            # 3. Rating (20% weight)
            rating_score = restaurant['avg_rating'] / 5.0
            score += 0.20 * rating_score
            
            # 4. Delivery time (15% weight)
            # Lower is better
            delivery_score = 1 - ((restaurant['avg_delivery_time'] - 20) / 40)
            delivery_score = max(0, min(1, delivery_score))
            score += 0.15 * delivery_score
            
            scores.append({
                'restaurant_id': restaurant['restaurant_id'],
                'content_score': score
            })
        
        recommendations = pd.DataFrame(scores)
        
        # Exclude restaurants from order history if provided
        if user_order_history:
            recommendations = recommendations[
                ~recommendations['restaurant_id'].isin(user_order_history)
            ]
        
        # Sort and return top-N
        recommendations = recommendations.sort_values(
            'content_score', ascending=False
        ).head(n_recommendations)
        
        # Normalize scores to 0-1 range
        if len(recommendations) > 0 and recommendations['content_score'].max() > 0:
            recommendations['content_score'] = (
                recommendations['content_score'] / recommendations['content_score'].max()
            )
        
        return recommendations.reset_index(drop=True)
    
    def _cold_start_recommend(self, n_recommendations: int) -> pd.DataFrame:
        """
        Fallback recommendations for new users
        Uses popularity and rating
        """
        # Sort by popularity and rating
        top_restaurants = self.restaurant_features.sort_values(
            by=['popularity_score', 'avg_rating'],
            ascending=False
        ).head(n_recommendations)
        
        recommendations = pd.DataFrame({
            'restaurant_id': top_restaurants['restaurant_id'],
            'content_score': np.linspace(1.0, 0.5, len(top_restaurants))  # Decreasing scores
        })
        
        return recommendations
    
    def save_model(self, filepath: str = None):
        """
        Save the fitted model
        """
        if filepath is None:
            filepath = MODELS_DIR / 'content_based_model.pkl'
        
        model_data = {
            'restaurant_vectors': self.restaurant_vectors,
            'scaler': self.scaler,
            'fitted': self.fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Saved content-based model to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str = None, 
                   restaurant_features: pd.DataFrame = None,
                   user_features: pd.DataFrame = None):
        """
        Load a saved model
        """
        if filepath is None:
            filepath = MODELS_DIR / 'content_based_model.pkl'
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Reconstruct model
        model = cls(restaurant_features, user_features)
        model.restaurant_vectors = model_data['restaurant_vectors']
        model.scaler = model_data['scaler']
        model.fitted = model_data['fitted']
        
        print(f"✅ Loaded content-based model from {filepath}")
        return model


if __name__ == "__main__":
    print("="*80)
    print(" CONTENT-BASED FILTERING MODEL")
    print("="*80)
    print()
    
    # Load features
    print("📂 Loading features...")
    restaurant_features = pd.read_csv(PROCESSED_DATA_DIR / 'restaurant_features.csv')
    user_features = pd.read_csv(PROCESSED_DATA_DIR / 'user_features.csv')
    
    # Train model
    print("\n🎯 Training content-based model...")
    cb_model = ContentBasedRecommender(restaurant_features, user_features)
    cb_model.fit()
    
    # Test recommendations
    sample_user = user_features['user_id'].iloc[0]
    print(f"\n🧪 Testing recommendations for user: {sample_user}")
    recommendations = cb_model.recommend(sample_user, n_recommendations=10)
    print(f"\nTop 10 Recommendations:")
    print(recommendations)
    
    # Save model
    cb_model.save_model()
    
    print("\n✅ Content-based filtering complete!")