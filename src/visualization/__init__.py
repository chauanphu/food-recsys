"""Visualization module for dish similarity analysis."""

from src.visualization.similarity import (
    compute_jaccard_matrix,
    compute_ingredient_embedding_matrix,
    compute_image_embedding_matrix,
)

__all__ = [
    "compute_jaccard_matrix",
    "compute_ingredient_embedding_matrix",
    "compute_image_embedding_matrix",
]
