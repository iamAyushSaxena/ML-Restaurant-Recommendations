"""
Explainability Engine
Generates human-readable explanations for recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from config import *


class ExplainabilityEngine:
    """
    Generates explanations for why restaurants were recommended
    """

    def __init__(
        self, restaurant_features: pd.DataFrame, user_features: pd.DataFrame, cf_model
    ):
        self.restaurant_features = restaurant_features
        self.user_features = user_features
        self.cf_model = cf_model

    def explain(
        self, user_id: str, restaurant_id: str, context: Optional[Dict] = None
    ) -> Dict:
        """
        Generate comprehensive explanation for a recommendation

        Returns:
            Dict with explanation components:
            - primary_reason: Main reason for recommendation
            - supporting_reasons: List of supporting factors
            - explanation_text: Full human-readable explanation
        """

        # Get restaurant and user data
        restaurant = self.restaurant_features[
            self.restaurant_features["restaurant_id"] == restaurant_id
        ].iloc[0]

        user = (
            self.user_features[self.user_features["user_id"] == user_id].iloc[0]
            if user_id in self.user_features["user_id"].values
            else None
        )

        reasons = []

        # === 1. User History Match ===
        if user is not None:
            user_history = self.cf_model.get_user_order_history(user_id)

            # Check if user ordered this cuisine before
            if len(user_history) > 0:
                # Get cuisines from order history
                historical_cuisines = self.restaurant_features[
                    self.restaurant_features["restaurant_id"].isin(user_history)
                ]["cuisine_type"].value_counts()

                if len(historical_cuisines) > 0:
                    top_cuisine = historical_cuisines.index[0]
                    cuisine_count = historical_cuisines.iloc[0]

                    if restaurant["cuisine_type"] == top_cuisine:
                        reasons.append(
                            {
                                "type": "user_history",
                                "weight": "high",
                                "text": EXPLANATION_TEMPLATES["user_history"].format(
                                    cuisine=top_cuisine, count=cuisine_count
                                ),
                            }
                        )

        # === 2. Similar Users ===
        if user_id in self.cf_model.interaction_matrix.index:
            similar_users = self.cf_model.get_similar_users(user_id, k=10)

            if similar_users:
                # Check if similar users ordered from this restaurant
                similar_user_ids = [uid for uid, _ in similar_users]
                similar_users_who_ordered = [
                    uid
                    for uid in similar_user_ids
                    if restaurant_id in self.cf_model.get_user_order_history(uid)
                ]

                if len(similar_users_who_ordered) >= 3:
                    reasons.append(
                        {
                            "type": "collaborative",
                            "weight": "high",
                            "text": f"Popular among users with similar taste ({len(similar_users_who_ordered)} similar users love this)",
                        }
                    )

        # === 3. High Rating ===
        if restaurant["avg_rating"] >= 4.0:
            reasons.append(
                {
                    "type": "quality",
                    "weight": "medium",
                    "text": EXPLANATION_TEMPLATES["highly_rated"].format(
                        rating=restaurant["avg_rating"],
                        reviews=restaurant["total_reviews"],
                    ),
                }
            )

        # === 4. Contextual Relevance ===
        if context:
            time_of_day = context.get("time_of_day")

            # Cuisine-time matching
            time_cuisine_match = {
                "breakfast": ["South Indian", "Cafe", "Beverages"],
                "lunch": ["North Indian", "South Indian", "Biryani"],
                "dinner": ["North Indian", "Biryani", "Chinese", "Continental"],
                "late_night": ["Fast Food", "Street Food", "Chinese"],
            }

            if time_of_day and restaurant["cuisine_type"] in time_cuisine_match.get(
                time_of_day, []
            ):
                meal_name = time_of_day.replace("_", " ").title()
                reasons.append(
                    {
                        "type": "contextual",
                        "weight": "medium",
                        "text": EXPLANATION_TEMPLATES["contextual"].format(
                            meal_time=meal_name
                        ),
                    }
                )

        # === 5. Proximity ===
        # Note: Would need user location to calculate actual distance
        if restaurant["avg_delivery_time"] <= 30:
            reasons.append(
                {
                    "type": "proximity",
                    "weight": "low",
                    "text": f"Quick delivery in ~{restaurant['avg_delivery_time']} minutes",
                }
            )

        # === 6. Value for Money ===
        if restaurant.get("value_score", 0) >= 1.2:  # High value score
            # CHANGED: Use actual price instead of symbols
            price_display = f"₹{int(restaurant['avg_order_value'])}"

            reasons.append(
                {
                    "type": "value",
                    "weight": "low",
                    "text": f"Great value for money ({price_display} with {restaurant['avg_rating']}★ rating)",
                }
            )

        # === 7. New Discovery ===
        if user is not None and user_id in self.cf_model.interaction_matrix.index:
            user_history = self.cf_model.get_user_order_history(user_id)
            if restaurant_id not in user_history:
                # Check if it's a new restaurant matching user's preferences
                if user.get("favorite_cuisine") == restaurant["cuisine_type"]:
                    reasons.append(
                        {
                            "type": "discovery",
                            "weight": "medium",
                            "text": EXPLANATION_TEMPLATES["new_discovery"],
                        }
                    )

        # === 8. Trending ===
        if restaurant.get("popularity_score", 0) >= 0.7:
            reasons.append(
                {
                    "type": "trending",
                    "weight": "low",
                    "text": EXPLANATION_TEMPLATES["trending"],
                }
            )

        # === Sort reasons by weight ===
        weight_order = {"high": 0, "medium": 1, "low": 2}
        reasons = sorted(reasons, key=lambda x: weight_order[x["weight"]])

        # === Generate Primary Reason ===
        primary_reason = reasons[0]["text"] if reasons else "Recommended for you"

        # === Generate Full Explanation ===
        if len(reasons) <= 1:
            explanation_text = primary_reason
        elif len(reasons) == 2:
            explanation_text = f"{primary_reason}. Also, {reasons[1]['text'].lower()}"
        else:
            supporting_text = " • ".join([r["text"] for r in reasons[1:3]])
            explanation_text = f"{primary_reason}. {supporting_text}"

        return {
            "restaurant_id": restaurant_id,
            "restaurant_name": restaurant["name"],
            "primary_reason": primary_reason,
            "supporting_reasons": [
                r["text"] for r in reasons[1:4]
            ],  # Top 3 supporting reasons
            "explanation_text": explanation_text,
            "all_reasons": reasons,
        }

    def batch_explain(
        self,
        user_id: str,
        recommendations: pd.DataFrame,
        context: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Generate explanations for multiple recommendations

        Args:
            user_id: User ID
            recommendations: DataFrame with restaurant_id column
            context: Contextual information

        Returns:
            List of explanation dictionaries
        """
        explanations = []

        for _, row in recommendations.iterrows():
            restaurant_id = row["restaurant_id"]
            explanation = self.explain(user_id, restaurant_id, context)
            explanations.append(explanation)

        return explanations


if __name__ == "__main__":
    print("=" * 80)
    print(" EXPLAINABILITY ENGINE")
    print("=" * 80)
    print()

    # Load data
    print("📂 Loading data...")
    restaurant_features = pd.read_csv(PROCESSED_DATA_DIR / "restaurant_features.csv")
    user_features = pd.read_csv(PROCESSED_DATA_DIR / "user_features.csv")
    interaction_matrix = pd.read_csv(
        PROCESSED_DATA_DIR / "interaction_matrix.csv", index_col=0
    )

    from collaborative_filtering import CollaborativeFilteringRecommender

    cf_model = CollaborativeFilteringRecommender.load_model()

    # Create explainer
    explainer = ExplainabilityEngine(restaurant_features, user_features, cf_model)

    # Test explanation
    sample_user = user_features["user_id"].iloc[10]
    sample_restaurant = restaurant_features["restaurant_id"].iloc[0]

    context = {"time_of_day": "dinner", "weather": "clear"}

    print(f"\n🧪 Testing explanation for:")
    print(f"   User: {sample_user}")
    print(f"   Restaurant: {sample_restaurant}")
    print(f"   Context: {context}")

    explanation = explainer.explain(sample_user, sample_restaurant, context)

    print(f"\n📝 Explanation:")
    print(f"   Primary Reason: {explanation['primary_reason']}")
    print(f"\n   Full Explanation:")
    print(f"   {explanation['explanation_text']}")

    if explanation["supporting_reasons"]:
        print(f"\n   Supporting Reasons:")
        for reason in explanation["supporting_reasons"]:
            print(f"   • {reason}")

    print("\n✅ Explainability engine ready!")
