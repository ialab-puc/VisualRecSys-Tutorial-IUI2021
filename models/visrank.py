import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# https://www.kaggle.com/jonathanoheix/product-recommendation-based-on-visual-similarity
class VisRank:

    def __init__(self, embedding, similarity_method=cosine_similarity):
        self.embedding = embedding
        self.similarities = pd.DataFrame(similarity_method(self.embedding))

    def most_similar_to_item(self, item, k=5):
        # Ignore first items (it's the same)
        closest_items = self.similarities[item].sort_values(ascending=False)[1:]
        if k is not None:
            # Select next k items (first is already ignored)
            closest_items = closest_items[:k]
        # Return index and score of similar items
        return np.array(closest_items.index), np.array(closest_items)

    def most_similar_to_profile(self, items, k=None, method="maximum", top=None, include_consumed=False):
        # Top argument is only needed in a specific method
        if method != "average_top_k" and top is not None:
            raise ValueError("top should be None unless method is 'average_top_k'")
        elif method == "average_top_k" and top is None:
            raise ValueError("top should not be None if method is 'average_top_k'")
        # Retrieve similarities of seen items towards all the items
        possible_items = self.similarities[items]
        if method == "maximum":
            # score(u, i) = max(sim(Vi, Vj) for j in P_u)
            score_ui = possible_items.max(axis=1)
        elif method == "average_top_k":
            # score(u, i) = largest(sim(Vi, Vj) for j in P_u, min(top, |P_u|)) / min(top, |P_u|)
            top = min(len(items), top)
            possible_items = possible_items.T
            possible_items = possible_items.nlargest(top, possible_items.columns)
            score_ui = possible_items.mean()
        elif method == "average":
            # score(u, i) = sum(sim(Vi, Vj) for j in P_u) / |P_u|
            score_ui = np.array(possible_items.mean(axis=1))
        else:
            raise ValueError("method has to be 'maximum', 'average_top_k' or 'average'")
        # Calculate score and retrieve relevant indexes
        score_ui = np.array(score_ui)
        # Retrieve relevant indexes
        recommendation = score_ui.argsort()[::-1]
        # Remove seen items indexes
        if not include_consumed:
            recommendation = np.delete(recommendation, np.where(np.isin(recommendation, np.array(items))))
        # If k is None, all items are calculated
        if k is not None:
            recommendation = recommendation[:k]
        return recommendation, score_ui[recommendation]


if __name__ == '__main__':
    embedding = np.random.rand(20, 100)
    print("Embedding size:", embedding.shape)
    model = VisRank(embedding, similarity_method=cosine_similarity)
    items = [0, 17, 3]
    print("Consumed items:", items)
    print("-" * 70)
    indexes, scores = model.most_similar_to_profile(items, k=10, method="maximum")
    print("Top items using maximum:\t\t", indexes)
    indexes, scores = model.most_similar_to_profile(items, k=10, method="average_top_k", top=2)
    print("Top items using average_top_k:\t", indexes)
    indexes, scores = model.most_similar_to_profile(items, k=10, method="average")
    print("Top items using average:\t\t", indexes)
