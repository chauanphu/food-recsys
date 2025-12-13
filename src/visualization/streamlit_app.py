"""Streamlit dashboard for dish similarity visualization.

Connects to Neo4j to fetch dish data and displays pairwise similarity
matrices using three different metrics: Jaccard, ingredient embeddings,
and image embeddings.

Run with: streamlit run src/visualization/streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to path to enable absolute imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

from src.services.neo4j_service import Neo4jService
from src.visualization.similarity import (
    compute_jaccard_matrix,
    compute_ingredient_embedding_matrix,
    compute_image_embedding_matrix,
)


# Page configuration
st.set_page_config(
    page_title="Dish Similarity Matrix",
    page_icon="🍽️",
    layout="wide",
)


@st.cache_resource
def get_neo4j_service() -> Neo4jService:
    """Get cached Neo4j service instance."""
    service = Neo4jService()
    service.verify_connectivity()
    return service


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_dishes_summary(_service: Neo4jService, limit: int) -> list[dict]:
    """Fetch dish summary from Neo4j with caching."""
    return _service.get_dishes_summary(limit=limit)


@st.cache_data(ttl=300)
def fetch_dishes_with_embeddings(
    _service: Neo4jService,
    dish_ids: tuple[str, ...],
) -> list[dict]:
    """Fetch dishes with embeddings from Neo4j with caching."""
    return _service.get_dishes_with_embeddings_and_ingredients(list(dish_ids))


def render_plotly_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
) -> go.Figure:
    """Render an interactive Plotly heatmap."""
    fig = px.imshow(
        matrix,
        x=labels,
        y=labels,
        color_continuous_scale="RdYlGn",
        aspect="equal",
        zmin=0,
        zmax=1,
        labels={"color": "Similarity"},
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis=dict(tickangle=45, side="bottom"),
        yaxis=dict(tickangle=0),
        width=800,
        height=800,
    )
    return fig


def render_seaborn_clustermap(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
) -> plt.Figure:
    """Render a hierarchical clustermap using seaborn."""
    # Create clustermap with hierarchical clustering
    g = sns.clustermap(
        matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        figsize=(12, 12),
        dendrogram_ratio=(0.1, 0.1),
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        method="average",  # Linkage method for hierarchical clustering
    )
    g.fig.suptitle(title, y=1.02, fontsize=14)
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(),
        rotation=45,
        ha="right",
        fontsize=8,
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(),
        rotation=0,
        fontsize=8,
    )
    return g.fig


def main():
    """Main Streamlit application."""
    st.title("🍽️ Dish Similarity Matrix")
    st.markdown(
        "Visualize pairwise similarity between dishes using different metrics."
    )

    # Initialize Neo4j connection
    try:
        neo4j_service = get_neo4j_service()
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}")
        st.info("Make sure Neo4j is running and check your connection settings.")
        return

    # Sidebar controls
    st.sidebar.header("⚙️ Settings")

    # Max dishes limit
    max_dishes = st.sidebar.number_input(
        "Maximum dishes to load",
        min_value=2,
        max_value=100,
        value=30,
        step=5,
        help="Limit the number of dishes loaded from the database",
    )

    # Fetch available dishes
    with st.spinner("Loading dishes from database..."):
        dishes_summary = fetch_dishes_summary(neo4j_service, limit=max_dishes)

    if not dishes_summary:
        st.warning("No dishes found with image embeddings in the database.")
        st.info("Please register some dishes first using the API.")
        return

    # Create dish name to ID mapping
    dish_options = {d["name"]: d["dish_id"] for d in dishes_summary}
    dish_names = list(dish_options.keys())

    st.sidebar.markdown(f"**Found {len(dish_names)} dishes**")

    # Multi-select for dishes
    selected_names = st.sidebar.multiselect(
        "Select dishes to compare",
        options=dish_names,
        default=dish_names[:min(10, len(dish_names))],
        help="Choose dishes to include in the similarity matrix",
    )

    if len(selected_names) < 2:
        st.warning("Please select at least 2 dishes to compare.")
        return

    # Metric selection
    metric = st.sidebar.radio(
        "Similarity Metric",
        options=["Jaccard (Ingredients)", "Ingredient Embeddings", "Image Embeddings"],
        index=0,
        help=(
            "**Jaccard**: Based on shared ingredients (set overlap)\n\n"
            "**Ingredient Embeddings**: Cosine similarity of averaged ingredient vectors\n\n"
            "**Image Embeddings**: Cosine similarity of dish images (visual similarity)"
        ),
    )

    # Visualization type toggle
    viz_type = st.sidebar.radio(
        "Visualization Type",
        options=["Heatmap (Plotly)", "Clustermap (Seaborn)"],
        index=0,
        help=(
            "**Heatmap**: Interactive, zoomable visualization\n\n"
            "**Clustermap**: Includes hierarchical clustering dendrograms"
        ),
    )

    # Fetch full dish data
    selected_ids = tuple(dish_options[name] for name in selected_names)

    with st.spinner("Fetching dish embeddings..."):
        dishes = fetch_dishes_with_embeddings(neo4j_service, selected_ids)

    if not dishes:
        st.error("Failed to fetch dish data.")
        return

    # Compute similarity matrix based on selected metric
    with st.spinner("Computing similarity matrix..."):
        if metric == "Jaccard (Ingredients)":
            matrix, labels = compute_jaccard_matrix(dishes)
            skipped = []
        elif metric == "Ingredient Embeddings":
            matrix, labels, skipped = compute_ingredient_embedding_matrix(dishes)
        else:  # Image Embeddings
            matrix, labels, skipped = compute_image_embedding_matrix(dishes)

    # Show warnings for skipped dishes
    if skipped:
        st.warning(
            f"⚠️ {len(skipped)} dish(es) skipped due to missing embeddings: "
            f"{', '.join(skipped[:5])}{'...' if len(skipped) > 5 else ''}"
        )

    if len(labels) < 2:
        st.error("Not enough dishes with valid data to compute similarity matrix.")
        return

    # Display matrix statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dishes Compared", len(labels))
    with col2:
        # Get upper triangle values (excluding diagonal)
        upper_tri = matrix[np.triu_indices(len(labels), k=1)]
        st.metric("Mean Similarity", f"{np.mean(upper_tri):.3f}")
    with col3:
        st.metric("Min Similarity", f"{np.min(upper_tri):.3f}")
    with col4:
        st.metric("Max Similarity", f"{np.max(upper_tri):.3f}")

    # Render visualization
    st.markdown("---")

    if viz_type == "Heatmap (Plotly)":
        title = f"Dish Similarity Matrix ({metric})"
        fig = render_plotly_heatmap(matrix, labels, title)
        st.plotly_chart(fig, use_container_width=True)
    else:
        title = f"Dish Similarity Clustermap ({metric})"
        fig = render_seaborn_clustermap(matrix, labels, title)
        st.pyplot(fig)
        plt.close(fig)

    # Show raw data in expander
    with st.expander("📊 View Raw Similarity Data"):
        import pandas as pd

        df = pd.DataFrame(matrix, index=labels, columns=labels)
        st.dataframe(df.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1))

    # Most similar pairs
    with st.expander("🔗 Most Similar Pairs"):
        pairs = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                pairs.append({
                    "Dish A": labels[i],
                    "Dish B": labels[j],
                    "Similarity": matrix[i, j],
                })
        import pandas as pd

        pairs_df = pd.DataFrame(pairs)
        pairs_df = pairs_df.sort_values("Similarity", ascending=False).head(20)
        st.dataframe(pairs_df, use_container_width=True)


if __name__ == "__main__":
    main()
