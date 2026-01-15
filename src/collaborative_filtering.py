"""
Collaborative Filtering Recommender
User-based collaborative filtering using cosine similarity
OPTIMIZED: Computes similarity on-the-fly to prevent MemoryError
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import pickle
from typing import List, Dict, Tuple
from config import *


class CollaborativeFilteringRecommender:
    """
    User-based collaborative filtering
    Recommends restaurants based on similar users' preferences
    """

    def __init__(self, interaction_matrix: pd.DataFrame):
        """
        Args:
            interaction_matrix: User-restaurant interaction scores (users × restaurants)
        """
        self.interaction_matrix = interaction_matrix
        # We NO LONGER store the full user_similarity_matrix to save RAM
        self.sparse_interaction_matrix = None
        self.fitted = False

    def fit(self):
        """
        Prepare sparse matrix for efficient on-the-fly calculation
        """
        print("🔧 Preparing sparse matrix for efficient similarity calculation...")

        # Convert to sparse matrix for efficiency
        # This uses very little memory compared to the dense matrix
        self.sparse_interaction_matrix = csr_matrix(self.interaction_matrix.values)

        self.fitted = True
        print(
            f"✅ Collaborative Filtering model fitted (Sparse Matrix Shape: {self.sparse_interaction_matrix.shape})"
        )

    def get_similar_users(self, user_id: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top-k most similar users (Calculated On-The-Fly)
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if user_id not in self.interaction_matrix.index:
            return []

        # Get the index of the target user
        user_idx = self.interaction_matrix.index.get_loc(user_id)

        # Get the target user's vector
        user_vector = self.sparse_interaction_matrix[user_idx]

        # Calculate cosine similarity between this user and ALL other users
        # This is fast! (Vector * Matrix multiplication)
        similarities = cosine_similarity(
            user_vector, self.sparse_interaction_matrix
        ).flatten()

        # Get indices of top k similar users (excluding the user themselves)
        # We get k+1 because the user is most similar to themselves
        similar_indices = similarities.argsort()[-(k + 1) :][::-1]

        similar_users = []
        for idx in similar_indices:
            if idx != user_idx:
                similar_user_id = self.interaction_matrix.index[idx]
                score = similarities[idx]
                if score > 0:
                    similar_users.append((similar_user_id, score))

        return similar_users[:k]

    def recommend(
        self,
        user_id: str,
        n_recommendations: int = 10,
        exclude_already_ordered: bool = True,
    ) -> pd.DataFrame:
        """
        Generate restaurant recommendations using collaborative filtering
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Handle new users (cold start)
        if user_id not in self.interaction_matrix.index:
            return pd.DataFrame(columns=["restaurant_id", "cf_score"])

        # Get similar users (On-the-fly calculation)
        similar_users = self.get_similar_users(user_id, k=30)

        if not similar_users:
            return pd.DataFrame(columns=["restaurant_id", "cf_score"])

        # Get restaurants the target user has already interacted with
        user_restaurants = self.interaction_matrix.loc[user_id]
        already_ordered = user_restaurants[user_restaurants > 0].index.tolist()

        # Aggregate scores from similar users
        restaurant_scores = {}
        total_similarity = sum(sim for _, sim in similar_users)

        for similar_user_id, similarity in similar_users:
            # Get this similar user's restaurant scores
            similar_user_scores = self.interaction_matrix.loc[similar_user_id]

            # Add weighted scores
            for restaurant_id, score in similar_user_scores.items():
                if score > 0:
                    if restaurant_id not in restaurant_scores:
                        restaurant_scores[restaurant_id] = 0
                    restaurant_scores[restaurant_id] += score * similarity

        # Normalize by total similarity
        if total_similarity > 0:
            restaurant_scores = {
                rid: score / total_similarity
                for rid, score in restaurant_scores.items()
            }

        # Exclude already ordered restaurants if requested
        if exclude_already_ordered:
            restaurant_scores = {
                rid: score
                for rid, score in restaurant_scores.items()
                if rid not in already_ordered
            }

        # Convert to DataFrame and sort
        recommendations = (
            pd.DataFrame(
                list(restaurant_scores.items()), columns=["restaurant_id", "cf_score"]
            )
            .sort_values("cf_score", ascending=False)
            .head(n_recommendations)
        )

        # Normalize scores to 0-1 range
        if len(recommendations) > 0 and recommendations["cf_score"].max() > 0:
            recommendations["cf_score"] = (
                recommendations["cf_score"] / recommendations["cf_score"].max()
            )

        return recommendations.reset_index(drop=True)

    def get_user_order_history(self, user_id: str) -> List[str]:
        """
        Get list of restaurants user has ordered from
        """
        if user_id not in self.interaction_matrix.index:
            return []

        user_restaurants = self.interaction_matrix.loc[user_id]
        return user_restaurants[user_restaurants > 0].index.tolist()

    def save_model(self, filepath: str = None):
        """
        Save the fitted model (LIGHTWEIGHT VERSION)
        """
        if filepath is None:
            filepath = MODELS_DIR / "collaborative_model.pkl"

        # Only save the interaction matrix and sparse matrix
        # NOT the huge similarity matrix
        model_data = {
            "interaction_matrix": self.interaction_matrix,
            "sparse_interaction_matrix": self.sparse_interaction_matrix,
            "fitted": self.fitted,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"💾 Saved collaborative filtering model to {filepath}")

    @classmethod
    def load_model(cls, filepath: str = None):
        """
        Load a saved model
        """
        if filepath is None:
            filepath = MODELS_DIR / "collaborative_model.pkl"

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        # Reconstruct model
        model = cls(model_data["interaction_matrix"])
        model.sparse_interaction_matrix = model_data.get("sparse_interaction_matrix")
        model.fitted = model_data["fitted"]

        # If loading an old model without sparse matrix, recreate it
        if model.sparse_interaction_matrix is None and model.fitted:
            model.sparse_interaction_matrix = csr_matrix(
                model.interaction_matrix.values
            )

        print(f"✅ Loaded collaborative filtering model from {filepath}")
        return model
