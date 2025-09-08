# app_book_cards_price.py
import streamlit as st
import pandas as pd
import joblib
from difflib import get_close_matches

# -------------------------
# Load dataset & similarity
# -------------------------
@st.cache_resource
def load_data():
    sim = joblib.load("artifacts/cosine_similarity_matrix.joblib")
    merged = pd.read_csv("artifacts/merged_books.csv")
    merged.columns = merged.columns.str.lower().str.strip()

    # Normalize titles
    merged["title"] = merged["title"].astype(str).str.strip().str.replace('"', '')

    # Clean Description
    if "description" in merged.columns:
        merged["description"] = merged["description"].fillna("No description available")
        merged["description"] = merged["description"].astype(str).str.replace(
            r"^Sorry.*", "No proper description available", regex=True
        )

    # Choose price column
    if "price_x" in merged.columns:
        merged["price"] = merged["price_x"]
    elif "price_y" in merged.columns:
        merged["price"] = merged["price_y"]
    else:
        merged["price"] = "N/A"

    return sim, merged

sim, merged = load_data()

# -------------------------
# Recommendation function
# -------------------------
def recommend_books(book_title, top_n=5):
    idx = merged[merged["title"] == book_title].index[0]
    scores = list(enumerate(sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recs = []
    for i, sc in scores[1:top_n+1]:  # skip itself
        recs.append({
            "Title": merged.loc[i, "title"],
            "Author": merged.loc[i, "author"] if "author" in merged.columns else "",
            "Price": merged.loc[i, "price"] if "price" in merged.columns else "N/A",
            "Rating": merged.loc[i, "rating"] if "rating" in merged.columns else "",
            "Description": merged.loc[i, "description"] if "description" in merged.columns else "",
            "Similarity": round(sc, 3)
        })
    return recs

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="📚 Book Recommender", layout="wide")

st.title("📚 Smart Book Recommender")
st.markdown("Find books similar to your favorite ones, powered by NLP & similarity search.")

# Search box
book_query = st.text_input("🔎 Search for a book title:")

if book_query:
    # Normalize query
    query_norm = book_query.strip().replace('"', '')

    # Try exact match
    if query_norm in merged["title"].values:
        selected_book = query_norm
    else:
        # Fuzzy match
        matches = get_close_matches(query_norm, merged["title"].tolist(), n=1, cutoff=0.6)
        selected_book = matches[0] if matches else None

    if selected_book:
        st.subheader(f"📖 Recommendations for **{selected_book}**")
        recs = recommend_books(selected_book, top_n=5)
        if recs:
            for rec in recs:
                with st.container():
                    st.markdown(f"### {rec['Title']}")
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"👤 **Author:** {rec['Author']}")
                        
                        st.markdown(f"⭐ **Rating:** {rec['Rating']}")
                        st.markdown(f"📖 **Description:** {rec['Description']}")
                    with col2:
                        st.metric(label="Similarity", value=rec["Similarity"])
                    st.markdown("---")
        else:
            st.warning("No recommendations found for this book.")
    else:
        st.error("❌ Book not found. Try another title.")
else:
    st.info("ℹ️ Start typing a book title above to see recommendations.")
