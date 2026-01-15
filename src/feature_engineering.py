"""
Feature Engineering for Restaurant Recommendations
Creates features for ML models
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.spatial.distance import cdist
from config import *


class FeatureEngineer:
    """
    Transforms raw data into ML-ready features
    """

    def __init__(
        self,
        users_df: pd.DataFrame,
        restaurants_df: pd.DataFrame,
        orders_df: pd.DataFrame,
    ):
        self.users_df = users_df
        self.restaurants_df = restaurants_df
        self.orders_df = orders_df
        self.user_features = None
        self.restaurant_features = None
        self.interaction_matrix = None

    def create_user_features(self) -> pd.DataFrame:
        """
        Create engineered features for users
        """
        print("🔧 Engineering user features...")

        # Start with base user features
        user_features = self.users_df.copy()

        # === Order-based features ===
        user_order_stats = (
            self.orders_df.groupby("user_id")
            .agg(
                {
                    "order_id": "count",
                    "order_value": ["mean", "std", "sum"],
                    "delivery_time": "mean",
                    "user_rating": "mean",
                    "restaurant_id": "nunique",  # Unique restaurants ordered from
                    "cuisine_type": lambda x: (
                        x.mode()[0] if not x.empty else None
                    ),  # Most frequent cuisine
                }
            )
            .reset_index()
        )

        user_order_stats.columns = [
            "user_id",
            "order_count",
            "avg_order_value",
            "order_value_std",
            "total_spent",
            "avg_delivery_time",
            "avg_rating_given",
            "unique_restaurants",
            "most_ordered_cuisine",
        ]

        # Merge with base features
        user_features = user_features.merge(user_order_stats, on="user_id", how="left")

        # Fill missing values for users with no orders
        user_features["order_count"] = user_features["order_count"].fillna(0)
        user_features["unique_restaurants"] = user_features[
            "unique_restaurants"
        ].fillna(0)

        # === Cuisine diversity score ===
        # How diverse are user's cuisine choices?
        cuisine_diversity = (
            self.orders_df.groupby("user_id")["cuisine_type"]
            .apply(lambda x: len(x.unique()) / len(x) if len(x) > 0 else 0)
            .reset_index()
        )
        cuisine_diversity.columns = ["user_id", "cuisine_diversity"]
        user_features = user_features.merge(cuisine_diversity, on="user_id", how="left")
        user_features["cuisine_diversity"] = user_features["cuisine_diversity"].fillna(
            0
        )

        # === Recency features ===
        # Days since first order
        user_first_order = (
            self.orders_df.groupby("user_id")["order_timestamp"].min().reset_index()
        )
        user_first_order["days_since_first_order"] = (
            datetime.now() - user_first_order["order_timestamp"]
        ).dt.days
        user_features = user_features.merge(
            user_first_order[["user_id", "days_since_first_order"]],
            on="user_id",
            how="left",
        )

        # === User segment (for easier grouping) ===
        user_features["user_segment"] = "casual"
        user_features.loc[user_features["order_count"] >= 20, "user_segment"] = (
            "regular"
        )
        user_features.loc[user_features["order_count"] >= 50, "user_segment"] = (
            "power_user"
        )

        # === Price preference score (1-4 scale) ===
        price_map = {"low": 1.5, "medium": 2.5, "high": 3.5}
        user_features["price_preference_score"] = user_features[
            "price_sensitivity"
        ].map(price_map)

        # === Encoding categorical variables ===
        # Label encode dietary preference
        dietary_map = {"veg": 1, "non_veg": 2, "vegan": 3, "no_preference": 0}
        user_features["dietary_preference_encoded"] = user_features[
            "dietary_preference"
        ].map(dietary_map)

        # Label encode favorite cuisine
        le_cuisine = LabelEncoder()
        user_features["favorite_cuisine_encoded"] = le_cuisine.fit_transform(
            user_features["favorite_cuisine"]
        )

        print(f"✅ Created {len(user_features.columns)} user features")
        self.user_features = user_features
        return user_features

    def create_restaurant_features(self) -> pd.DataFrame:
        """
        Create engineered features for restaurants
        """
        print("🔧 Engineering restaurant features...")

        restaurant_features = self.restaurants_df.copy()

        # === Order-based restaurant features ===
        restaurant_order_stats = (
            self.orders_df.groupby("restaurant_id")
            .agg(
                {
                    "order_id": "count",
                    "order_value": ["mean", "sum"],
                    "user_rating": "mean",
                    "user_id": "nunique",  # Unique customers
                }
            )
            .reset_index()
        )

        restaurant_order_stats.columns = [
            "restaurant_id",
            "total_orders",
            "avg_order_value",
            "total_revenue",
            "avg_user_rating",
            "unique_customers",
        ]

        restaurant_features = restaurant_features.merge(
            restaurant_order_stats, on="restaurant_id", how="left"
        )

        # Fill missing values for restaurants with no orders
        restaurant_features["total_orders"] = restaurant_features[
            "total_orders"
        ].fillna(0)
        restaurant_features["unique_customers"] = restaurant_features[
            "unique_customers"
        ].fillna(0)

        # === Popularity score (normalized) ===
        # Combination of orders, reviews, and rating
        max_orders = restaurant_features["total_orders"].max() or 1
        max_reviews = restaurant_features["total_reviews"].max() or 1

        restaurant_features["popularity_score"] = (
            0.4 * (restaurant_features["total_orders"] / max_orders)
            + 0.3 * (restaurant_features["total_reviews"] / max_reviews)
            + 0.3 * (restaurant_features["avg_rating"] / 5.0)
        )

        # === Customer retention rate ===
        # Customers who ordered more than once from this restaurant
        repeat_customers = (
            self.orders_df.groupby(["restaurant_id", "user_id"])
            .size()
            .reset_index(name="order_count")
        )
        repeat_customers = repeat_customers[repeat_customers["order_count"] > 1]
        repeat_stats = (
            repeat_customers.groupby("restaurant_id")
            .size()
            .reset_index(name="repeat_customers")
        )

        restaurant_features = restaurant_features.merge(
            repeat_stats, on="restaurant_id", how="left"
        )
        restaurant_features["repeat_customers"] = restaurant_features[
            "repeat_customers"
        ].fillna(0)

        restaurant_features["retention_rate"] = restaurant_features[
            "repeat_customers"
        ] / restaurant_features["unique_customers"].replace(0, 1)

        # === Delivery efficiency score ===
        # Lower delivery time = higher score
        restaurant_features["delivery_efficiency"] = 1 - (
            (restaurant_features["avg_delivery_time"] - 20) / 40
        )
        restaurant_features["delivery_efficiency"] = restaurant_features[
            "delivery_efficiency"
        ].clip(0, 1)

        # === Price-to-rating ratio (value for money) ===
        restaurant_features["value_score"] = (
            restaurant_features["avg_rating"] / restaurant_features["price_range"]
        )

        # === Restaurant segment ===
        restaurant_features["restaurant_segment"] = "emerging"
        restaurant_features.loc[
            restaurant_features["total_orders"] >= 100, "restaurant_segment"
        ] = "established"
        restaurant_features.loc[
            restaurant_features["total_orders"] >= 500, "restaurant_segment"
        ] = "popular"

        # === Encoding categorical variables ===
        le_cuisine = LabelEncoder()
        restaurant_features["cuisine_type_encoded"] = le_cuisine.fit_transform(
            restaurant_features["cuisine_type"]
        )

        # Veg indicator (already boolean, convert to int)
        restaurant_features["is_veg_only_int"] = restaurant_features[
            "is_veg_only"
        ].astype(int)

        print(f"✅ Created {len(restaurant_features.columns)} restaurant features")
        self.restaurant_features = restaurant_features
        return restaurant_features

    def create_user_restaurant_matrix(self) -> pd.DataFrame:
        """
        Create user-restaurant interaction matrix for collaborative filtering
        Format: rows=users, columns=restaurants, values=interaction_score
        """
        print("🔧 Creating user-restaurant interaction matrix...")

        # Calculate interaction score based on:
        # - Number of orders (frequency)
        # - Recency of orders
        # - User rating

        interactions = self.orders_df.copy()

        # Frequency score
        user_restaurant_orders = (
            interactions.groupby(["user_id", "restaurant_id"])
            .agg({"order_id": "count", "user_rating": "mean", "order_timestamp": "max"})
            .reset_index()
        )

        user_restaurant_orders.columns = [
            "user_id",
            "restaurant_id",
            "order_count",
            "avg_rating",
            "last_order",
        ]

        # Recency score (exponential decay)
        days_since_last = (
            datetime.now() - user_restaurant_orders["last_order"]
        ).dt.days
        user_restaurant_orders["recency_score"] = np.exp(
            -days_since_last / 30
        )  # 30-day half-life

        # Combined interaction score
        user_restaurant_orders["interaction_score"] = (
            0.4
            * np.log1p(user_restaurant_orders["order_count"])  # Log scale for frequency
            + 0.3 * (user_restaurant_orders["avg_rating"] / 5.0)
            + 0.3 * user_restaurant_orders["recency_score"]
        )

        # Pivot to matrix format
        interaction_matrix = user_restaurant_orders.pivot(
            index="user_id", columns="restaurant_id", values="interaction_score"
        ).fillna(0)

        print(
            f"✅ Created interaction matrix: {interaction_matrix.shape[0]} users × {interaction_matrix.shape[1]} restaurants"
        )
        print(
            f"   Sparsity: {(interaction_matrix == 0).sum().sum() / (interaction_matrix.shape[0] * interaction_matrix.shape[1]) * 100:.1f}%"
        )

        self.interaction_matrix = interaction_matrix
        return interaction_matrix

    def calculate_distance_features(
        self, user_location: Tuple[float, float]
    ) -> pd.DataFrame:
        """
        Calculate distance from user to each restaurant

        Args:
            user_location: (latitude, longitude) tuple

        Returns:
            DataFrame with restaurant_id and distance_km
        """
        user_lat, user_lon = user_location

        # Calculate haversine distance
        restaurant_coords = self.restaurant_features[
            ["restaurant_id", "location_lat", "location_lon"]
        ].copy()

        # Simplified distance calculation (not exact haversine)
        # For demo purposes, using Euclidean distance with lat/lon scaling
        lat_diff = (
            restaurant_coords["location_lat"] - user_lat
        ) * 111  # 1 degree lat ≈ 111 km
        lon_diff = (
            (restaurant_coords["location_lon"] - user_lon)
            * 111
            * np.cos(np.radians(user_lat))
        )

        restaurant_coords["distance_km"] = np.sqrt(lat_diff**2 + lon_diff**2)

        return restaurant_coords[["restaurant_id", "distance_km"]]

    def create_contextual_features(self, context: Dict) -> Dict:
        """
        Create features based on current context

        Args:
            context: Dict with keys like 'time_of_day', 'day_of_week', 'weather'

        Returns:
            Dict of contextual feature values
        """
        contextual_features = {}

        # Time of day encoding
        time_map = {
            "breakfast": [1, 0, 0, 0, 0],
            "lunch": [0, 1, 0, 0, 0],
            "evening_snack": [0, 0, 1, 0, 0],
            "dinner": [0, 0, 0, 1, 0],
            "late_night": [0, 0, 0, 0, 1],
        }
        time_of_day = context.get("time_of_day", "lunch")
        contextual_features["time_encoding"] = time_map.get(
            time_of_day, [0, 1, 0, 0, 0]
        )

        # Day of week (binary: weekday vs weekend)
        contextual_features["is_weekend"] = (
            1 if context.get("day_of_week") == "weekend" else 0
        )

        # Weather influence on cuisine preference
        weather = context.get("weather", "clear")
        weather_cuisine_boost = {
            "rainy": {
                "Street Food": 0.5,
                "Fast Food": 1.5,
                "Chinese": 1.3,
            },  # Comfort food
            "hot": {
                "Beverages": 1.8,
                "Desserts": 1.5,
                "South Indian": 0.8,
            },  # Cool items
            "clear": {},  # No adjustment
        }
        contextual_features["weather_cuisine_boost"] = weather_cuisine_boost.get(
            weather, {}
        )

        return contextual_features

    def save_features(self):
        """
        Save engineered features to files
        """
        if self.user_features is not None:
            self.user_features.to_csv(
                PROCESSED_DATA_DIR / "user_features.csv", index=False
            )
            print(
                f"💾 Saved user features to {PROCESSED_DATA_DIR / 'user_features.csv'}"
            )

        if self.restaurant_features is not None:
            self.restaurant_features.to_csv(
                PROCESSED_DATA_DIR / "restaurant_features.csv", index=False
            )
            print(
                f"💾 Saved restaurant features to {PROCESSED_DATA_DIR / 'restaurant_features.csv'}"
            )

        if self.interaction_matrix is not None:
            self.interaction_matrix.to_csv(
                PROCESSED_DATA_DIR / "interaction_matrix.csv"
            )
            print(
                f"💾 Saved interaction matrix to {PROCESSED_DATA_DIR / 'interaction_matrix.csv'}"
            )


if __name__ == "__main__":
    print("=" * 80)
    print(" FEATURE ENGINEERING")
    print("=" * 80)
    print()

    # Load data
    print("📂 Loading data...")
    users_df = pd.read_csv(SYNTHETIC_DATA_DIR / "users.csv")
    restaurants_df = pd.read_csv(SYNTHETIC_DATA_DIR / "restaurants.csv")
    orders_df = pd.read_csv(SYNTHETIC_DATA_DIR / "orders.csv")
    orders_df["order_timestamp"] = pd.to_datetime(orders_df["order_timestamp"])

    # Engineer features
    engineer = FeatureEngineer(users_df, restaurants_df, orders_df)

    user_features = engineer.create_user_features()
    restaurant_features = engineer.create_restaurant_features()
    interaction_matrix = engineer.create_user_restaurant_matrix()

    # Save features
    engineer.save_features()

    print("\n✅ Feature engineering complete!")
