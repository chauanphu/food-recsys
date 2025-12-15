"""Streamlit page for dish recommendations.

Displays recommendations using both Collaborative Filtering and Content-Based Filtering.
"""

import streamlit as st
import pandas as pd
import requests
from typing import Any

def api_get_users(api_base_url: str) -> list[dict]:
    """Fetch all users."""
    try:
        resp = requests.get(f"{api_base_url.rstrip('/')}/users", params={"limit": 100}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("users", [])
    except Exception as e:
        st.error(f"Failed to fetch users: {e}")
        return []

def api_get_recommendations(api_base_url: str, user_id: str, method: str) -> dict:
    """Fetch recommendations for a user."""
    try:
        resp = requests.get(
            f"{api_base_url.rstrip('/')}/users/{user_id}/recommendations",
            params={"limit": 10, "method": method},
            timeout=20
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Failed to fetch recommendations ({method}): {e}")
        return {}

def api_get_similar_users(api_base_url: str, user_id: str) -> dict:
    """Fetch similar users."""
    try:
        resp = requests.get(
            f"{api_base_url.rstrip('/')}/users/{user_id}/similar",
            params={"limit": 5},
            timeout=20
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        # Don't show error for similar users as it might be empty/irrelevant for content-based
        return {}

def api_search_by_image(api_base_url: str, image_file: Any) -> dict:
    """Search for dishes by image."""
    try:
        files = {"image": (image_file.name, image_file, image_file.type)}
        resp = requests.post(
            f"{api_base_url.rstrip('/')}/dishes/search-by-image",
            files=files,
            params={"limit": 5, "threshold": 0.6},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Search failed: {e}")
        return {}

def render_recommendation_card(dish: dict, method: str):
    """Render a single recommendation card."""
    with st.container(border=True):
        st.subheader(dish["name"])
        
        col1, col2 = st.columns([1, 2])
        with col1:
            score_label = "Predicted Rating" if method != "content_based" else "Jaccard Score"
            st.metric(score_label, f"{dish['predicted_score']:.2f}")
            
            if method != "content_based":
                st.caption(f"Based on {dish['recommender_count']} similar users")
        
        with col2:
            if dish.get("description"):
                st.write(dish["description"])
            
            if dish.get("ingredients"):
                st.caption(f"**Ingredients:** {', '.join(dish['ingredients'][:5])}...")

def render_recommendations_page(api_base_url: str = "https://api.foodrecys.captechvn.com/api/v1") -> None:
    """Render the recommendation page."""
    st.title("🍽️ Dish Recommendations")
    
    tab1, tab2 = st.tabs(["User Recommendations", "Image Search"])
    
    with tab1:
        st.markdown("Compare **Collaborative Filtering** (User-based) vs **Content-Based Filtering** (Ingredient-based).")

        # User Selection
        users = api_get_users(api_base_url)
        if not users:
            st.warning("No users found. Please ingest users first.")
        else:
            user_options = {f"{u['name']} ({u['user_id']})": u['user_id'] for u in users}
            selected_user_label = st.selectbox("Select User", options=list(user_options.keys()))
            
            if selected_user_label:
                selected_user_id = user_options[selected_user_label]
                
                # Fetch Data
                if st.button("Get Recommendations"):
                    col_cf, col_cb = st.columns(2)
                    
                    # Collaborative Filtering Column
                    with col_cf:
                        st.header("Collaborative Filtering")
                        st.caption("Recommends dishes liked by similar users.")
                        
                        with st.spinner("Fetching CF recommendations..."):
                            cf_data = api_get_recommendations(api_base_url, selected_user_id, "collaborative")
                            
                        if cf_data:
                            method_used = cf_data.get("method")
                            if method_used == "popular_fallback":
                                st.warning("⚠️ Not enough data for collaborative filtering. Showing popular dishes instead.")
                            
                            recs = cf_data.get("recommendations", [])
                            if not recs:
                                st.info("No recommendations found.")
                            else:
                                for dish in recs:
                                    render_recommendation_card(dish, "collaborative")
                                    
                            # Show similar users
                            if method_used == "collaborative_filtering":
                                with st.expander("👥 Similar Users"):
                                    sim_data = api_get_similar_users(api_base_url, selected_user_id)
                                    if sim_data and sim_data.get("similar_users"):
                                        sim_users = sim_data["similar_users"]
                                        df = pd.DataFrame(sim_users)
                                        st.dataframe(
                                            df[["name", "similarity", "shared_dishes"]],
                                            hide_index=True,
                                            use_container_width=True
                                        )

                    # Content-Based Filtering Column
                    with col_cb:
                        st.header("Content-Based Filtering")
                        st.caption("Recommends dishes with ingredients similar to what you like.")
                        
                        with st.spinner("Fetching CB recommendations..."):
                            cb_data = api_get_recommendations(api_base_url, selected_user_id, "content_based")
                            
                        if cb_data:
                            recs = cb_data.get("recommendations", [])
                            if not recs:
                                st.info("No recommendations found. Try rating more dishes with diverse ingredients.")
                            else:
                                for dish in recs:
                                    render_recommendation_card(dish, "content_based")

    with tab2:
        st.header("📷 Search by Image")
        st.caption("Upload a food image to find similar dishes and their ingredients.")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", width=300)
            
            if st.button("Search Similar Dishes"):
                with st.spinner("Searching..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    data = api_search_by_image(api_base_url, uploaded_file)
                    
                if data and data.get("results"):
                    st.success(f"Found {data['count']} similar dishes!")
                    
                    for result in data["results"]:
                        with st.container(border=True):
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.metric("Similarity", f"{result['score']:.2f}")
                                    
                            with col2:
                                st.subheader(result["name"])
                                if result.get("description"):
                                    st.write(result["description"])
                                
                                if result.get("ingredients"):
                                    st.markdown("**Ingredients:**")
                                    st.write(", ".join(result["ingredients"]))
                else:
                    st.info("No similar dishes found.")
