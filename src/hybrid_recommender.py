"""
Hybrid Recommendation System
Combines collaborative filtering, content-based filtering, and contextual factors
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import pickle
from config import *
from collaborative_filtering import CollaborativeFilteringRecommender
from content_based_filtering import ContentBasedRecommender


class HybridRecommender:
    """
    Hybrid recommendation system that combines multiple approaches
    """

    def __init__(
        self,
        cf_model: CollaborativeFilteringRecommender,
        cb_model: ContentBasedRecommender,
        restaurant_features: pd.DataFrame,
        user_features: pd.DataFrame,
    ):
        """
        Args:
            cf_model: Trained collaborative filtering model
            cb_model: Trained content-based model
            restaurant_features: Restaurant data
            user_features: User data
        """
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.restaurant_features = restaurant_features
        self.user_features = user_features
        self.weights = MODEL_WEIGHTS.copy()

    def recommend(
        self,
        user_id: str,
        n_recommendations: int = 10,
        context: Optional[Dict] = None,
        user_location: Optional[Tuple[float, float]] = None,
        exclude_ordered: bool = True,
    ) -> pd.DataFrame:
        """
        Generate hybrid recommendations

        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations
            context: Contextual information (time, weather, etc.)
            user_location: (latitude, longitude) tuple
            exclude_ordered: Exclude already ordered restaurants

        Returns:
            DataFrame with restaurant_id, final_score, and component scores
        """

        # Get user order history
        user_order_history = self.cf_model.get_user_order_history(user_id)
        has_history = len(user_order_history) >= MIN_ORDERS_FOR_CF

        # === 1. Collaborative Filtering Scores ===
        if has_history:
            cf_recommendations = self.cf_model.recommend(
                user_id,
                n_recommendations=50,  # Get more candidates
                exclude_already_ordered=exclude_ordered,
            )
        else:
            # Not enough history for CF
            cf_recommendations = pd.DataFrame(columns=["restaurant_id", "cf_score"])

        # === 2. Content-Based Scores ===
        cb_recommendations = self.cb_model.recommend(
            user_id,
            n_recommendations=50,
            user_order_history=user_order_history if exclude_ordered else None,
        )

        # === 3. Merge Scores ===
        # Start with all unique restaurants from both models
        all_restaurants = (
            pd.concat(
                [
                    (
                        cf_recommendations[["restaurant_id"]]
                        if len(cf_recommendations) > 0
                        else pd.DataFrame(columns=["restaurant_id"])
                    ),
                    cb_recommendations[["restaurant_id"]],
                ]
            )
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # Merge CF scores
        if len(cf_recommendations) > 0:
            all_restaurants = all_restaurants.merge(
                cf_recommendations[["restaurant_id", "cf_score"]],
                on="restaurant_id",
                how="left",
            )
        else:
            all_restaurants["cf_score"] = 0.0

        # Merge CB scores
        all_restaurants = all_restaurants.merge(
            cb_recommendations[["restaurant_id", "content_score"]],
            on="restaurant_id",
            how="left",
        )

        # Fill missing scores with 0
        all_restaurants["cf_score"] = all_restaurants["cf_score"].fillna(0)
        all_restaurants["content_score"] = all_restaurants["content_score"].fillna(0)

        # === 4. Add Restaurant Features ===
        all_restaurants = all_restaurants.merge(
            self.restaurant_features[
                [
                    "restaurant_id",
                    "name",
                    "cuisine_type",
                    "avg_rating",
                    "price_range",
                    "avg_delivery_time",
                    "popularity_score",
                    "location_lat",
                    "location_lon",
                    "avg_order_value",
                ]
            ],
            on="restaurant_id",
            how="left",
        )

        # === 5. Contextual Scoring ===
        contextual_score = self._compute_contextual_score(
            all_restaurants, context, user_location
        )
        all_restaurants["contextual_score"] = contextual_score

        # === 6. Compute Final Hybrid Score ===
        # Adjust weights if user has no CF history
        if not has_history:
            # Shift CF weight to CB
            adjusted_weights = {
                "collaborative_filtering": 0.0,
                "content_based": self.weights["collaborative_filtering"]
                + self.weights["content_based"],
                "contextual": self.weights["contextual"],
            }
        else:
            adjusted_weights = self.weights

        all_restaurants["final_score"] = (
            adjusted_weights["collaborative_filtering"] * all_restaurants["cf_score"]
            + adjusted_weights["content_based"] * all_restaurants["content_score"]
            + adjusted_weights["contextual"] * all_restaurants["contextual_score"]
        )

        # === 7. Apply Business Rules & Filters ===
        # Filter out restaurants with very low ratings
        all_restaurants = all_restaurants[all_restaurants["avg_rating"] >= 3.0]

        # Filter by distance if user location provided
        if user_location:
            all_restaurants = self._filter_by_distance(
                all_restaurants, user_location, max_distance_km=10
            )

        # === 8. Rank and Return Top-N ===
        recommendations = all_restaurants.sort_values(
            "final_score", ascending=False
        ).head(n_recommendations)

        # Add rank
        recommendations["rank"] = range(1, len(recommendations) + 1)

        # Select columns to return
        result_columns = [
            "rank",
            "restaurant_id",
            "name",
            "cuisine_type",
            "avg_rating",
            "price_range",
            "avg_delivery_time",
            "final_score",
            "cf_score",
            "content_score",
            "contextual_score",
            "avg_order_value",
        ]

        return recommendations[result_columns].reset_index(drop=True)

    def _compute_contextual_score(
        self,
        restaurants: pd.DataFrame,
        context: Optional[Dict],
        user_location: Optional[Tuple[float, float]],
    ) -> np.ndarray:
        """
        Compute contextual relevance score
        """
        scores = np.ones(len(restaurants))  # Start with neutral score

        if context is None:
            return scores

        # === Time of Day Context ===
        time_of_day = context.get("time_of_day", "lunch")

        # Cuisine preferences by time
        time_cuisine_boost = {
            "breakfast": {
                "South Indian": 1.3,
                "Cafe": 1.4,
                "Fast Food": 1.2,
                "Beverages": 1.3,
            },
            "lunch": {
                "North Indian": 1.2,
                "South Indian": 1.2,
                "Biryani": 1.3,
                "Chinese": 1.1,
            },
            "dinner": {
                "North Indian": 1.3,
                "Biryani": 1.4,
                "Chinese": 1.2,
                "Continental": 1.1,
            },
            "late_night": {"Fast Food": 1.5, "Street Food": 1.4, "Chinese": 1.2},
        }

        boost_map = time_cuisine_boost.get(time_of_day, {})
        for cuisine, boost in boost_map.items():
            scores[restaurants["cuisine_type"] == cuisine] *= boost

        # === Weather Context ===
        weather = context.get("weather", "clear")
        weather_boost = {
            "rainy": {
                "Street Food": 0.6,  # Penalize street food in rain
                "Fast Food": 1.3,  # Boost comfort food
                "Chinese": 1.2,
            },
            "hot": {"Beverages": 1.5, "Desserts": 1.3, "South Indian": 1.1},
        }

        boost_map = weather_boost.get(weather, {})
        for cuisine, boost in boost_map.items():
            scores[restaurants["cuisine_type"] == cuisine] *= boost

        # === Distance Penalty (if location provided) ===
        if user_location:
            distances = self._calculate_distances(restaurants, user_location)
            # Exponential decay: closer is much better
            distance_scores = np.exp(-distances / 3.0)  # 3km half-life
            scores *= distance_scores

        # === Popularity Boost ===
        # Popular restaurants get slight boost
        scores *= 1 + 0.2 * restaurants["popularity_score"].values

        # Normalize to 0-1 range
        if scores.max() > 0:
            scores = scores / scores.max()

        return scores

    def _calculate_distances(
        self, restaurants: pd.DataFrame, user_location: Tuple[float, float]
    ) -> np.ndarray:
        """
        Calculate distances from user to restaurants (in km)
        """
        user_lat, user_lon = user_location

        # Simplified distance calculation
        lat_diff = (restaurants["location_lat"].values - user_lat) * 111
        lon_diff = (
            (restaurants["location_lon"].values - user_lon)
            * 111
            * np.cos(np.radians(user_lat))
        )

        distances = np.sqrt(lat_diff**2 + lon_diff**2)
        return distances

    def _filter_by_distance(
        self,
        restaurants: pd.DataFrame,
        user_location: Tuple[float, float],
        max_distance_km: float = 10,
    ) -> pd.DataFrame:
        """
        Filter restaurants by maximum distance
        """
        distances = self._calculate_distances(restaurants, user_location)
        restaurants["distance_km"] = distances

        return restaurants[restaurants["distance_km"] <= max_distance_km]

    def explain_recommendation(
        self, user_id: str, restaurant_id: str, context: Optional[Dict] = None
    ) -> Dict:
        """
        Generate explanation for why a restaurant was recommended
        Returns dict with explanation components
        """
        from explainability import ExplainabilityEngine

        explainer = ExplainabilityEngine(
            self.restaurant_features, self.user_features, self.cf_model
        )

        return explainer.explain(user_id, restaurant_id, context)

    def save_model(self, filepath: str = None):
        """
        Save the hybrid model configuration
        """
        if filepath is None:
            filepath = MODELS_DIR / "hybrid_model.pkl"

        model_data = {"weights": self.weights}

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"💾 Saved hybrid model configuration to {filepath}")

    @classmethod
    def load_model(
        cls,
        cf_model: CollaborativeFilteringRecommender,
        cb_model: ContentBasedRecommender,
        restaurant_features: pd.DataFrame,
        user_features: pd.DataFrame,
        filepath: str = None,
    ):
        """
        Load a saved hybrid model
        """
        if filepath is None:
            filepath = MODELS_DIR / "hybrid_model.pkl"

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        model = cls(cf_model, cb_model, restaurant_features, user_features)
        model.weights = model_data["weights"]

        print(f"✅ Loaded hybrid model from {filepath}")
        return model


if __name__ == "__main__":
    print("=" * 80)
    print(" HYBRID RECOMMENDATION SYSTEM")
    print("=" * 80)
    print()

    # Load all components
    print("📂 Loading models and data...")

    # Load data
    restaurant_features = pd.read_csv(PROCESSED_DATA_DIR / "restaurant_features.csv")
    user_features = pd.read_csv(PROCESSED_DATA_DIR / "user_features.csv")
    interaction_matrix = pd.read_csv(
        PROCESSED_DATA_DIR / "interaction_matrix.csv", index_col=0
    )

    # Load models
    cf_model = CollaborativeFilteringRecommender.load_model()
    cb_model = ContentBasedRecommender.load_model(
        restaurant_features=restaurant_features, user_features=user_features
    )

    # Create hybrid model
    print("\n🎯 Creating hybrid recommender...")
    hybrid_model = HybridRecommender(
        cf_model, cb_model, restaurant_features, user_features
    )

    # Test recommendations
    sample_user = user_features["user_id"].iloc[10]
    sample_location = (28.5, 77.1)  # Delhi coordinates

    context = {"time_of_day": "dinner", "day_of_week": "weekend", "weather": "clear"}

    print(f"\n🧪 Testing recommendations for user: {sample_user}")
    print(f"   Context: {context}")
    print(f"   Location: {sample_location}")

    recommendations = hybrid_model.recommend(
        user_id=sample_user,
        n_recommendations=10,
        context=context,
        user_location=sample_location,
    )

    print(f"\n📋 Top 10 Recommendations:")
    print(
        recommendations[["rank", "name", "cuisine_type", "avg_rating", "final_score"]]
    )

    # Save model
    hybrid_model.save_model()

    print("\n✅ Hybrid recommendation system ready!")
