"""
Cold Start Handler
Handles recommendations for new users with no order history
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from config import *


class ColdStartHandler:
    """
    Handles recommendations for users with insufficient data
    """

    def __init__(self, restaurant_features: pd.DataFrame):
        self.restaurant_features = restaurant_features

    def onboarding_recommend(
        self, preferences: Dict, n_recommendations: int = 10
    ) -> pd.DataFrame:
        """
        Generate recommendations based on onboarding questionnaire responses

        Args:
            preferences: Dict with keys:
                - dietary_preference: 'veg', 'non_veg', 'vegan', 'no_preference'
                - favorite_cuisines: List of cuisine types
                - budget: '₹0-200', '₹200-400', '₹400-600', '₹600+'
            n_recommendations: Number of recommendations

        Returns:
            DataFrame with restaurant recommendations
        """

        candidates = self.restaurant_features.copy()

        # === Apply Filters ===

        # 1. Dietary filter
        dietary_pref = preferences.get("dietary_preference", "no_preference")
        if dietary_pref in ["veg", "vegan"]:
            candidates = candidates[candidates["is_veg_only"] == True]

        # 2. Budget filter
        budget = preferences.get("budget", "₹200-400")
        budget_map = {
            "₹0-200": [1],
            "₹200-400": [1, 2],
            "₹400-600": [2, 3],
            "₹600+": [3, 4],
        }
        allowed_price_ranges = budget_map.get(budget, [1, 2])
        candidates = candidates[candidates["price_range"].isin(allowed_price_ranges)]

        # === Calculate Scores ===

        candidates["cold_start_score"] = 0.0

        # 1. Cuisine match (50% weight)
        favorite_cuisines = preferences.get("favorite_cuisines", [])
        if favorite_cuisines:
            for cuisine in favorite_cuisines:
                candidates.loc[
                    candidates["cuisine_type"] == cuisine, "cold_start_score"
                ] += 0.50 / len(favorite_cuisines)

        # 2. Popularity (30% weight)
        max_popularity = candidates["popularity_score"].max() or 1
        candidates["cold_start_score"] += 0.30 * (
            candidates["popularity_score"] / max_popularity
        )

        # 3. Rating (20% weight)
        candidates["cold_start_score"] += 0.20 * (candidates["avg_rating"] / 5.0)

        # === Ensure Diversity ===
        # Don't show all restaurants from same cuisine
        recommendations = []
        cuisine_counts = {}

        sorted_candidates = candidates.sort_values("cold_start_score", ascending=False)

        for _, restaurant in sorted_candidates.iterrows():
            cuisine = restaurant["cuisine_type"]

            # Limit to max 3 restaurants per cuisine
            if cuisine_counts.get(cuisine, 0) < 3:
                recommendations.append(restaurant)
                cuisine_counts[cuisine] = cuisine_counts.get(cuisine, 0) + 1

            if len(recommendations) >= n_recommendations:
                break

        # If not enough diverse recommendations, add remaining top ones
        if len(recommendations) < n_recommendations:
            remaining = sorted_candidates.head(n_recommendations).to_dict("records")
            for rest in remaining:
                if len(recommendations) >= n_recommendations:
                    break
                if rest["restaurant_id"] not in [
                    r["restaurant_id"] for r in recommendations
                ]:
                    recommendations.append(rest)

        result = pd.DataFrame(recommendations)

        # Add rank
        result["rank"] = range(1, len(result) + 1)

        # Select columns
        result_columns = [
            "rank",
            "restaurant_id",
            "name",
            "cuisine_type",
            "avg_rating",
            "price_range",
            "avg_delivery_time",
            "cold_start_score",
            "avg_order_value",
        ]

        return result[result_columns].reset_index(drop=True)

    def popular_recommend(self, n_recommendations: int = 10) -> pd.DataFrame:
        """
        Fallback: Recommend most popular restaurants
        Used when no preferences are provided
        """

        # Sort by popularity and rating
        popular = (
            self.restaurant_features.sort_values(
                by=["popularity_score", "avg_rating", "total_reviews"], ascending=False
            )
            .head(n_recommendations)
            .copy()
        )

        popular["rank"] = range(1, len(popular) + 1)

        result_columns = [
            "rank",
            "restaurant_id",
            "name",
            "cuisine_type",
            "avg_rating",
            "price_range",
            "avg_delivery_time",
            "popularity_score",
            "avg_order_value",
        ]

        return popular[result_columns].reset_index(drop=True)

    def similar_user_cold_start(
        self,
        user_profile: Dict,
        all_users: pd.DataFrame,
        interaction_matrix: pd.DataFrame,
        n_recommendations: int = 10,
    ) -> pd.DataFrame:
        """
        Find similar users based on profile and use their preferences
        (When user has profile but no orders yet)
        """

        # Find users with similar profile
        similar_users = []

        favorite_cuisine = user_profile.get("favorite_cuisine")
        dietary_pref = user_profile.get("dietary_preference")
        price_sensitivity = user_profile.get("price_sensitivity")

        for _, other_user in all_users.iterrows():
            similarity_score = 0

            # Cuisine match
            if other_user["favorite_cuisine"] == favorite_cuisine:
                similarity_score += 0.5

            # Dietary match
            if other_user["dietary_preference"] == dietary_pref:
                similarity_score += 0.3

            # Price sensitivity match
            if other_user["price_sensitivity"] == price_sensitivity:
                similarity_score += 0.2

            if similarity_score > 0.5:  # Threshold
                similar_users.append((other_user["user_id"], similarity_score))

        if not similar_users:
            # Fallback to popular
            return self.popular_recommend(n_recommendations)

        # Get restaurants these similar users ordered from
        restaurant_scores = {}

        for similar_user_id, similarity in similar_users[:20]:  # Top 20 similar users
            if similar_user_id in interaction_matrix.index:
                user_restaurants = interaction_matrix.loc[similar_user_id]

                for restaurant_id, score in user_restaurants.items():
                    if score > 0:
                        if restaurant_id not in restaurant_scores:
                            restaurant_scores[restaurant_id] = 0
                        restaurant_scores[restaurant_id] += score * similarity

        # Sort and get top-N
        top_restaurants = sorted(
            restaurant_scores.items(), key=lambda x: x[1], reverse=True
        )[:n_recommendations]

        # Get restaurant details
        top_restaurant_ids = [rid for rid, _ in top_restaurants]

        result = self.restaurant_features[
            self.restaurant_features["restaurant_id"].isin(top_restaurant_ids)
        ].copy()

        # Add scores
        score_map = dict(top_restaurants)
        result["cold_start_score"] = result["restaurant_id"].map(score_map)

        # Sort by score
        result = result.sort_values("cold_start_score", ascending=False)
        result["rank"] = range(1, len(result) + 1)

        result_columns = [
            "rank",
            "restaurant_id",
            "name",
            "cuisine_type",
            "avg_rating",
            "price_range",
            "avg_delivery_time",
            "cold_start_score",
            "avg_order_value",
        ]

        return result[result_columns].reset_index(drop=True)


if __name__ == "__main__":
    print("=" * 80)
    print(" COLD START HANDLER")
    print("=" * 80)
    print()

    # Load data
    print("📂 Loading data...")
    restaurant_features = pd.read_csv(PROCESSED_DATA_DIR / "restaurant_features.csv")

    # Create handler
    handler = ColdStartHandler(restaurant_features)

    # Test onboarding recommendations
    print("\n🧪 Testing onboarding recommendations...")
    preferences = {
        "dietary_preference": "veg",
        "favorite_cuisines": ["South Indian", "North Indian"],
        "budget": "₹200-400",
    }

    recommendations = handler.onboarding_recommend(preferences, n_recommendations=10)

    print(f"\nTop 10 Recommendations for new user:")
    print(
        recommendations[
            ["rank", "name", "cuisine_type", "avg_rating", "cold_start_score"]
        ]
    )

    # Test popular recommendations
    print("\n🧪 Testing popular recommendations (fallback)...")
    popular = handler.popular_recommend(n_recommendations=10)
    print(f"\nTop 10 Popular Restaurants:")
    print(popular[["rank", "name", "cuisine_type", "avg_rating"]])

    print("\n✅ Cold start handler ready!")
