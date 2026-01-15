"""
Synthetic Data Generator
Creates realistic restaurant recommendation dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from config import *

np.random.seed(42)


class RestaurantDataGenerator:
    """
    Generates synthetic data for restaurant recommendation system
    """

    def __init__(self):
        self.users_df = None
        self.restaurants_df = None
        self.orders_df = None

    def generate_restaurants(self) -> pd.DataFrame:
        """
        Generate restaurant profiles with attributes
        """
        print(f"🍽️  Generating {RESTAURANT_BASE} restaurants...")

        restaurant_ids = [f"rest_{i:04d}" for i in range(RESTAURANT_BASE)]

        # Restaurant names (simplified)
        name_prefixes = [
            "Tasty",
            "Spicy",
            "Royal",
            "Golden",
            "Fresh",
            "Express",
            "Paradise",
            "Corner",
            "Deluxe",
            "Classic",
        ]
        name_suffixes = [
            "Kitchen",
            "Bites",
            "Hub",
            "Palace",
            "Point",
            "Cafe",
            "Restaurant",
            "Grill",
            "House",
            "Eatery",
        ]

        names = [
            f"{np.random.choice(name_prefixes)} {np.random.choice(name_suffixes)}"
            for _ in range(RESTAURANT_BASE)
        ]

        # Cuisine distribution
        cuisines = np.random.choice(
            CUISINE_TYPES,
            size=RESTAURANT_BASE,
            p=[0.20, 0.18, 0.15, 0.12, 0.08, 0.10, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01],
        )

        # Ratings (skewed towards higher ratings)
        ratings = np.random.beta(8, 2, RESTAURANT_BASE) * 2 + 3  # Range: 3-5
        ratings = np.clip(ratings, 3.0, 5.0)

        # Reviews count (log-normal distribution)
        reviews = np.random.lognormal(mean=5, sigma=1.5, size=RESTAURANT_BASE).astype(
            int
        )
        reviews = np.clip(reviews, 10, 10000)

        # Price range (1-4)
        price_ranges = np.random.choice(
            [1, 2, 3, 4], size=RESTAURANT_BASE, p=[0.15, 0.45, 0.30, 0.10]
        )

        # Delivery time (20-60 minutes)
        delivery_times = np.random.normal(35, 10, RESTAURANT_BASE)
        delivery_times = np.clip(delivery_times, 20, 60).astype(int)

        # Veg/Non-veg
        is_veg_only = np.random.choice(
            [True, False], size=RESTAURANT_BASE, p=[0.30, 0.70]
        )

        # Location (random coordinates in city grid)
        # Assume city bounds: lat 28.4-28.7, lon 77.0-77.3 (Delhi example)
        locations_lat = np.random.uniform(28.4, 28.7, RESTAURANT_BASE)
        locations_lon = np.random.uniform(77.0, 77.3, RESTAURANT_BASE)

        # Commission rate (platform's cut: 15-25%)
        commission_rates = np.random.uniform(0.15, 0.25, RESTAURANT_BASE)

        # Operating hours (simplified)
        operating_hours = np.random.choice(
            ["10AM-11PM", "11AM-12AM", "9AM-10PM", "24 hours"],
            size=RESTAURANT_BASE,
            p=[0.50, 0.30, 0.15, 0.05],
        )

        self.restaurants_df = pd.DataFrame(
            {
                "restaurant_id": restaurant_ids,
                "name": names,
                "cuisine_type": cuisines,
                "avg_rating": np.round(ratings, 1),
                "total_reviews": reviews,
                "price_range": price_ranges,
                "avg_delivery_time": delivery_times,
                "is_veg_only": is_veg_only,
                "location_lat": np.round(locations_lat, 4),
                "location_lon": np.round(locations_lon, 4),
                "operating_hours": operating_hours,
                "commission_rate": np.round(commission_rates, 2),
            }
        )

        print(f"✅ Generated {RESTAURANT_BASE} restaurants")
        return self.restaurants_df

    def generate_users(self) -> pd.DataFrame:
        """
        Generate user profiles
        """
        print(f"👥 Generating {USER_BASE} users...")

        user_ids = [f"user_{i:06d}" for i in range(USER_BASE)]

        # Total orders per user (some power users, many casual users)
        # Log-normal distribution
        total_orders = np.random.lognormal(mean=1.5, sigma=1.2, size=USER_BASE).astype(
            int
        )
        total_orders = np.clip(total_orders, 1, 200)

        # Average order value
        avg_order_values = np.random.gamma(shape=4, scale=100, size=USER_BASE)
        avg_order_values = np.clip(avg_order_values, 150, 1500)

        # Favorite cuisine (users have preferences)
        favorite_cuisines = np.random.choice(CUISINE_TYPES, size=USER_BASE)

        # Price sensitivity
        price_sensitivity = np.random.choice(
            ["low", "medium", "high"], size=USER_BASE, p=[0.25, 0.50, 0.25]
        )

        # Average rating given by user
        avg_ratings_given = np.random.beta(7, 2, USER_BASE) * 2 + 3
        avg_ratings_given = np.clip(avg_ratings_given, 3.0, 5.0)

        # Dietary preference
        dietary_preferences = np.random.choice(
            ["veg", "non_veg", "vegan", "no_preference"],
            size=USER_BASE,
            p=[0.30, 0.50, 0.05, 0.15],
        )

        # Preferred meal time
        preferred_meal_times = np.random.choice(
            ["breakfast", "lunch", "dinner", "snacks"],
            size=USER_BASE,
            p=[0.10, 0.35, 0.45, 0.10],
        )

        # User location
        locations_lat = np.random.uniform(28.4, 28.7, USER_BASE)
        locations_lon = np.random.uniform(77.0, 77.3, USER_BASE)

        # Days since last order (recency)
        days_since_last_order = np.random.exponential(scale=10, size=USER_BASE).astype(
            int
        )
        days_since_last_order = np.clip(days_since_last_order, 0, 90)

        self.users_df = pd.DataFrame(
            {
                "user_id": user_ids,
                "total_orders": total_orders,
                "avg_order_value": np.round(avg_order_values, 2),
                "favorite_cuisine": favorite_cuisines,
                "price_sensitivity": price_sensitivity,
                "avg_rating_given": np.round(avg_ratings_given, 1),
                "dietary_preference": dietary_preferences,
                "preferred_meal_time": preferred_meal_times,
                "location_lat": np.round(locations_lat, 4),
                "location_lon": np.round(locations_lon, 4),
                "days_since_last_order": days_since_last_order,
            }
        )

        print(f"✅ Generated {USER_BASE} users")
        return self.users_df

    def generate_orders(self) -> pd.DataFrame:
        """
        Generate historical order data (user-restaurant interactions)
        """
        if self.users_df is None or self.restaurants_df is None:
            raise ValueError("Generate users and restaurants first")

        print(f"📦 Generating {HISTORICAL_ORDERS} orders...")

        orders = []
        order_id = 0

        # Generate orders for each user based on their total_orders
        for idx, user in self.users_df.iterrows():
            user_id = user["user_id"]
            n_orders = user["total_orders"]

            # User preferences influence restaurant selection
            favorite_cuisine = user["favorite_cuisine"]
            dietary_pref = user["dietary_preference"]
            price_sensitivity = user["price_sensitivity"]

            # Filter restaurants based on preferences
            candidate_restaurants = self.restaurants_df.copy()

            # Dietary filtering
            if dietary_pref == "veg":
                candidate_restaurants = candidate_restaurants[
                    candidate_restaurants["is_veg_only"] == True
                ]
            elif dietary_pref == "vegan":
                candidate_restaurants = candidate_restaurants[
                    (candidate_restaurants["is_veg_only"] == True)
                    & (
                        candidate_restaurants["cuisine_type"].isin(
                            ["Healthy", "Salads"]
                        )
                    )
                ]

            # If too few candidates, relax constraints
            if len(candidate_restaurants) < 10:
                candidate_restaurants = self.restaurants_df.copy()

            # Price filtering based on sensitivity
            if price_sensitivity == "low":
                # Low sensitivity = high budget, prefer expensive restaurants
                candidate_restaurants = candidate_restaurants[
                    candidate_restaurants["price_range"] >= 2
                ]
            elif price_sensitivity == "high":
                # High sensitivity = low budget, prefer cheap restaurants
                candidate_restaurants = candidate_restaurants[
                    candidate_restaurants["price_range"] <= 2
                ]

            # Calculate probabilities for restaurant selection
            # Higher weight for favorite cuisine
            candidate_restaurants["selection_prob"] = 1.0
            candidate_restaurants.loc[
                candidate_restaurants["cuisine_type"] == favorite_cuisine,
                "selection_prob",
            ] *= 3.0  # 3x more likely to order favorite cuisine

            # Higher rated restaurants more likely
            candidate_restaurants["selection_prob"] *= (
                candidate_restaurants["avg_rating"] / 5.0
            )

            # Normalize probabilities
            candidate_restaurants["selection_prob"] /= candidate_restaurants[
                "selection_prob"
            ].sum()

            # Select restaurants for this user's orders
            # Users tend to repeat orders from liked restaurants
            unique_restaurants_count = max(
                3, int(n_orders * 0.6)
            )  # 60% unique restaurants
            selected_restaurants = np.random.choice(
                candidate_restaurants["restaurant_id"].values,
                size=min(unique_restaurants_count, len(candidate_restaurants)),
                replace=False,
                p=candidate_restaurants["selection_prob"].values,
            )

            # Generate orders with repetition
            for _ in range(n_orders):
                # 70% chance to repeat from selected restaurants
                if np.random.random() < 0.70 and len(selected_restaurants) > 0:
                    restaurant_id = np.random.choice(selected_restaurants)
                else:
                    # 30% chance to try new restaurant
                    restaurant_id = np.random.choice(
                        candidate_restaurants["restaurant_id"].values,
                        p=candidate_restaurants["selection_prob"].values,
                    )

                # Order details
                restaurant = self.restaurants_df[
                    self.restaurants_df["restaurant_id"] == restaurant_id
                ].iloc[0]

                # Order value based on user's avg and restaurant's price range
                base_value = user["avg_order_value"]
                price_multiplier = restaurant["price_range"] / 2.0
                order_value = (
                    base_value * price_multiplier * np.random.uniform(0.8, 1.2)
                )
                order_value = np.round(order_value, 2)

                # Order timestamp (random over past 6 months)
                days_ago = np.random.randint(0, 180)
                order_timestamp = datetime.now() - timedelta(days=days_ago)

                # Delivery time (close to restaurant's avg)
                delivery_time = restaurant["avg_delivery_time"] + np.random.randint(
                    -5, 10
                )
                delivery_time = max(15, delivery_time)

                # User rating (close to their avg)
                rating = user["avg_rating_given"] + np.random.uniform(-0.5, 0.5)
                rating = np.clip(rating, 1, 5)

                orders.append(
                    {
                        "order_id": f"order_{order_id:08d}",
                        "user_id": user_id,
                        "restaurant_id": restaurant_id,
                        "order_value": order_value,
                        "order_timestamp": order_timestamp,
                        "delivery_time": delivery_time,
                        "user_rating": np.round(rating, 1),
                        "cuisine_type": restaurant["cuisine_type"],
                        "price_range": restaurant["price_range"],
                    }
                )

                order_id += 1

                if order_id % 25000 == 0:
                    print(f"   Generated {order_id:,} orders...")

        self.orders_df = pd.DataFrame(orders)

        # Sort by timestamp
        self.orders_df = self.orders_df.sort_values("order_timestamp").reset_index(
            drop=True
        )

        print(f"✅ Generated {len(self.orders_df):,} orders")
        return self.orders_df

    def save_data(self):
        """
        Save all generated data to CSV files
        """
        if self.users_df is not None:
            self.users_df.to_csv(SYNTHETIC_DATA_DIR / "users.csv", index=False)
            print(f"💾 Saved users to {SYNTHETIC_DATA_DIR / 'users.csv'}")

        if self.restaurants_df is not None:
            self.restaurants_df.to_csv(
                SYNTHETIC_DATA_DIR / "restaurants.csv", index=False
            )
            print(f"💾 Saved restaurants to {SYNTHETIC_DATA_DIR / 'restaurants.csv'}")

        if self.orders_df is not None:
            self.orders_df.to_csv(SYNTHETIC_DATA_DIR / "orders.csv", index=False)
            print(f"💾 Saved orders to {SYNTHETIC_DATA_DIR / 'orders.csv'}")

    def generate_summary_stats(self):
        """
        Print summary statistics of generated data
        """
        print("\n" + "=" * 80)
        print("DATA GENERATION SUMMARY")
        print("=" * 80)

        if self.users_df is not None:
            print(f"\n👥 USERS ({len(self.users_df):,}):")
            print(f"   Total Orders Distribution:")
            print(f"   - Mean: {self.users_df['total_orders'].mean():.1f}")
            print(f"   - Median: {self.users_df['total_orders'].median():.1f}")
            print(f"   - Max: {self.users_df['total_orders'].max()}")
            print(f"\n   Dietary Preferences:")
            print(self.users_df["dietary_preference"].value_counts())

        if self.restaurants_df is not None:
            print(f"\n🍽️  RESTAURANTS ({len(self.restaurants_df):,}):")
            print(f"   Average Rating: {self.restaurants_df['avg_rating'].mean():.2f}")
            print(f"\n   Cuisine Distribution:")
            print(self.restaurants_df["cuisine_type"].value_counts().head(10))

        if self.orders_df is not None:
            print(f"\n📦 ORDERS ({len(self.orders_df):,}):")
            print(
                f"   Average Order Value: ₹{self.orders_df['order_value'].mean():.2f}"
            )
            print(
                f"   Average Delivery Time: {self.orders_df['delivery_time'].mean():.0f} min"
            )
            print(
                f"   Date Range: {self.orders_df['order_timestamp'].min().date()} to {self.orders_df['order_timestamp'].max().date()}"
            )


if __name__ == "__main__":
    print("=" * 80)
    print(" RESTAURANT RECOMMENDATION DATA GENERATOR")
    print("=" * 80)
    print()

    generator = RestaurantDataGenerator()

    # Generate data
    generator.generate_restaurants()
    generator.generate_users()
    generator.generate_orders()

    # Save data
    generator.save_data()

    # Print summary
    generator.generate_summary_stats()

    print("\n✅ Data generation complete!")
