"""
Model Evaluation Module
Evaluates recommendation quality using multiple metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import *


class RecommendationEvaluator:
    """
    Evaluates recommendation system performance
    """
    
    def __init__(self, 
                 test_orders: pd.DataFrame,
                 restaurant_features: pd.DataFrame):
        """
        Args:
            test_orders: Hold-out test set of actual orders
            restaurant_features: Restaurant metadata
        """
        self.test_orders = test_orders
        self.restaurant_features = restaurant_features
        
    def precision_at_k(self, 
                       recommendations: List[str],
                       actual_orders: List[str],
                       k: int = 10) -> float:
        """
        Precision@K: Proportion of recommended items that are relevant
        
        Args:
            recommendations: List of recommended restaurant_ids
            actual_orders: List of restaurants user actually ordered from
            k: Top-k recommendations to consider
            
        Returns:
            Precision score (0-1)
        """
        top_k_recs = recommendations[:k]
        
        # Count how many recommended restaurants were actually ordered
        hits = len(set(top_k_recs) & set(actual_orders))
        
        precision = hits / k if k > 0 else 0
        return precision
    
    def recall_at_k(self,
                    recommendations: List[str],
                    actual_orders: List[str],
                    k: int = 10) -> float:
        """
        Recall@K: Proportion of relevant items that are recommended
        
        Args:
            recommendations: List of recommended restaurant_ids
            actual_orders: List of restaurants user actually ordered from
            k: Top-k recommendations to consider
            
        Returns:
            Recall score (0-1)
        """
        top_k_recs = recommendations[:k]
        
        # Count how many actual orders were captured in recommendations
        hits = len(set(top_k_recs) & set(actual_orders))
        
        recall = hits / len(actual_orders) if len(actual_orders) > 0 else 0
        return recall
    
    def hit_rate_at_k(self,
                      recommendations: List[str],
                      actual_orders: List[str],
                      k: int = 10) -> float:
        """
        Hit Rate@K: Binary - did user order from any of top-k recommendations?
        
        Returns:
            1.0 if hit, 0.0 if miss
        """
        top_k_recs = recommendations[:k]
        
        # Check if there's any overlap
        hit = 1.0 if len(set(top_k_recs) & set(actual_orders)) > 0 else 0.0
        return hit
    
    def ndcg_at_k(self,
                  recommendations: List[str],
                  actual_orders: List[str],
                  k: int = 10) -> float:
        """
        Normalized Discounted Cumulative Gain@K
        Accounts for position of relevant items in recommendation list
        
        Returns:
            NDCG score (0-1)
        """
        top_k_recs = recommendations[:k]
        
        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, restaurant_id in enumerate(top_k_recs):
            if restaurant_id in actual_orders:
                # Relevance = 1 if ordered, 0 otherwise
                # Discount by position (1-indexed)
                dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0
        
        # Calculate IDCG (Ideal DCG)
        # Best case: all relevant items at top
        n_relevant = min(len(actual_orders), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
        
        # Normalize
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return ndcg
    
    def diversity_score(self, recommendations: List[str]) -> float:
        """
        Cuisine diversity in recommendations
        Higher diversity = better discovery
        
        Returns:
            Diversity score (0-1)
        """
        if len(recommendations) == 0:
            return 0.0
        
        # Get cuisines for recommended restaurants
        recommended_restaurants = self.restaurant_features[
            self.restaurant_features['restaurant_id'].isin(recommendations)
        ]
        
        cuisines = recommended_restaurants['cuisine_type'].values
        unique_cuisines = len(set(cuisines))
        
        # Normalize by total cuisines available
        max_diversity = min(len(cuisines), len(CUISINE_TYPES))
        diversity = unique_cuisines / max_diversity if max_diversity > 0 else 0
        
        return diversity
    
    def novelty_score(self,
                      recommendations: List[str],
                      user_order_history: List[str]) -> float:
        """
        Novelty: How many new restaurants are recommended?
        
        Returns:
            Proportion of new restaurants (0-1)
        """
        if len(recommendations) == 0:
            return 0.0
        
        # Count restaurants user hasn't tried before
        new_restaurants = [r for r in recommendations if r not in user_order_history]
        
        novelty = len(new_restaurants) / len(recommendations)
        return novelty
    
    def coverage_score(self,
                      all_recommendations: List[List[str]]) -> float:
        """
        Catalog Coverage: What % of restaurants are ever recommended?
        Measures if recommender is too narrow
        
        Args:
            all_recommendations: List of recommendation lists for multiple users
            
        Returns:
            Coverage score (0-1)
        """
        # Flatten all recommendations
        all_recommended = set()
        for recs in all_recommendations:
            all_recommended.update(recs)
        
        # Calculate coverage
        total_restaurants = len(self.restaurant_features)
        coverage = len(all_recommended) / total_restaurants if total_restaurants > 0 else 0
        
        return coverage
    
    def evaluate_model(self,
                       recommender,
                       test_users: List[str],
                       k_values: List[int] = [5, 10, 20]) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            recommender: Fitted recommendation model
            test_users: List of user_ids to test on
            k_values: List of k values for @k metrics
            
        Returns:
            Dict with all evaluation metrics
        """
        print(f"📊 Evaluating model on {len(test_users)} test users...")
        
        metrics = {
            'precision': {k: [] for k in k_values},
            'recall': {k: [] for k in k_values},
            'hit_rate': {k: [] for k in k_values},
            'ndcg': {k: [] for k in k_values},
            'diversity': [],
            'novelty': []
        }
        
        all_recommendations = []
        
        for user_id in test_users:
            # Get actual orders for this user in test set
            actual_orders = self.test_orders[
                self.test_orders['user_id'] == user_id
            ]['restaurant_id'].unique().tolist()
            
            if len(actual_orders) == 0:
                continue  # Skip users with no test orders
            
            # Get user's historical orders (before test period)
            user_history = recommender.cf_model.get_user_order_history(user_id)
            
            # Generate recommendations
            try:
                recommendations_df = recommender.recommend(
                    user_id=user_id,
                    n_recommendations=max(k_values),
                    exclude_ordered=True
                )
                
                recommended_restaurants = recommendations_df['restaurant_id'].tolist()
                all_recommendations.append(recommended_restaurants)
                
            except Exception as e:
                print(f"   Error generating recommendations for {user_id}: {e}")
                continue
            
            # Calculate metrics for each k
            for k in k_values:
                metrics['precision'][k].append(
                    self.precision_at_k(recommended_restaurants, actual_orders, k)
                )
                metrics['recall'][k].append(
                    self.recall_at_k(recommended_restaurants, actual_orders, k)
                )
                metrics['hit_rate'][k].append(
                    self.hit_rate_at_k(recommended_restaurants, actual_orders, k)
                )
                metrics['ndcg'][k].append(
                    self.ndcg_at_k(recommended_restaurants, actual_orders, k)
                )
            
            # Diversity and novelty (calculated once per user)
            metrics['diversity'].append(
                self.diversity_score(recommended_restaurants[:10])
            )
            metrics['novelty'].append(
                self.novelty_score(recommended_restaurants[:10], user_history)
            )
        
        # Calculate coverage
        coverage = self.coverage_score(all_recommendations)
        
        # Aggregate results
        results = {
            'precision@k': {k: np.mean(metrics['precision'][k]) for k in k_values},
            'recall@k': {k: np.mean(metrics['recall'][k]) for k in k_values},
            'hit_rate@k': {k: np.mean(metrics['hit_rate'][k]) for k in k_values},
            'ndcg@k': {k: np.mean(metrics['ndcg'][k]) for k in k_values},
            'diversity': np.mean(metrics['diversity']),
            'novelty': np.mean(metrics['novelty']),
            'coverage': coverage,
            'n_users_evaluated': len(test_users)
        }
        
        return results
    
    def print_evaluation_report(self, results: Dict):
        """
        Print formatted evaluation report
        """
        print("\n" + "="*80)
        print(" MODEL EVALUATION REPORT")
        print("="*80)
        print(f"\nEvaluated on {results['n_users_evaluated']} users")
        
        print("\n📈 ACCURACY METRICS:")
        print(f"   Precision@5:  {results['precision@k'][5]:.4f}")
        print(f"   Precision@10: {results['precision@k'][10]:.4f}")
        print(f"   Precision@20: {results['precision@k'][20]:.4f}")
        
        print(f"\n   Recall@5:     {results['recall@k'][5]:.4f}")
        print(f"   Recall@10:    {results['recall@k'][10]:.4f}")
        print(f"   Recall@20:    {results['recall@k'][20]:.4f}")
        
        print(f"\n   Hit Rate@5:   {results['hit_rate@k'][5]:.4f}")
        print(f"   Hit Rate@10:  {results['hit_rate@k'][10]:.4f}")
        print(f"   Hit Rate@20:  {results['hit_rate@k'][20]:.4f}")
        
        print(f"\n   NDCG@5:       {results['ndcg@k'][5]:.4f}")
        print(f"   NDCG@10:      {results['ndcg@k'][10]:.4f}")
        print(f"   NDCG@20:      {results['ndcg@k'][20]:.4f}")
        
        print("\n🎯 DISCOVERY METRICS:")
        print(f"   Diversity:    {results['diversity']:.4f}")
        print(f"   Novelty:      {results['novelty']:.4f}")
        print(f"   Coverage:     {results['coverage']:.4f}")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    print("="*80)
    print(" MODEL EVALUATION")
    print("="*80)
    print()
    
    # Load data
    print("📂 Loading data...")
    orders_df = pd.read_csv(SYNTHETIC_DATA_DIR / 'orders.csv')
    orders_df['order_timestamp'] = pd.to_datetime(orders_df['order_timestamp'])
    restaurant_features = pd.read_csv(PROCESSED_DATA_DIR / 'restaurant_features.csv')
    user_features = pd.read_csv(PROCESSED_DATA_DIR / 'user_features.csv')
    
    # Split data: last 20% of orders as test set
    orders_df = orders_df.sort_values('order_timestamp')
    split_idx = int(len(orders_df) * 0.8)
    train_orders = orders_df[:split_idx]
    test_orders = orders_df[split_idx:]
    
    print(f"   Training orders: {len(train_orders):,}")
    print(f"   Test orders: {len(test_orders):,}")
    
    # Load models
    from hybrid_recommender import HybridRecommender
    from collaborative_filtering import CollaborativeFilteringRecommender
    from content_based_filtering import ContentBasedRecommender
    
    interaction_matrix = pd.read_csv(PROCESSED_DATA_DIR / 'interaction_matrix.csv', index_col=0)
    cf_model = CollaborativeFilteringRecommender.load_model()
    cb_model = ContentBasedRecommender.load_model(
        restaurant_features=restaurant_features,
        user_features=user_features
    )
    hybrid_model = HybridRecommender(cf_model, cb_model, restaurant_features, user_features)
    
    # Evaluate
    evaluator = RecommendationEvaluator(test_orders, restaurant_features)
    
    # Get test users (users who have orders in test set)
    test_users = test_orders['user_id'].unique()[:100]  # Sample 100 users for faster evaluation
    
    results = evaluator.evaluate_model(
        hybrid_model,
        test_users.tolist(),
        k_values=[5, 10, 20]
    )
    
    # Print report
    evaluator.print_evaluation_report(results)
    
    print("\n✅ Evaluation complete!")