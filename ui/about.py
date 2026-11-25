"""Page 3 — About Model UI."""

import sys
from pathlib import Path
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def render_about():
    st.markdown(
        """
        <div class="hero">

        <h1>🧠 How the Model Works</h1>

        <p>
        Multimodal neural architecture combining user preferences,
        movie identity, poster information and trailer-audio information.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ARCHITECTURE

    st.header("🏗️ Multimodal Model Architecture")

    st.write("The final multimodal model combines four inputs:")

    st.code(
        """
                                    INPUTS
                                        │
        ┌──────────────┬───────────────┼───────────────┐
        │              │               │               │              
        ▼              ▼               ▼               ▼              
    User ID       Movie ID      Poster Image    Trailer Audio        
        │              │               │               │             
        ▼              ▼               ▼               ▼             
    Embedding(32)  Embedding(32)    CLIP 512-D       VGGish 128-D    
        │              │               │               │             
        │              │               ▼               ▼             
        │              │           Dense(32)       Dense(32)         
        │              │             ReLU            ReLU            
        │              │               │               │             
        └──────────────┴───────────────┴───────────────┘             
                                    │
                                    ▼
                                Concatenate
                                    │
                                    ▼
                                Dense(64)
                                  ReLU
                                    │
                                    ▼
                                Dropout(0.2)
                                    │
                                    ▼
                                Dense(32)
                                  ReLU
                                    │
                                    ▼
                                Dense(1)
                                    │
                                    ▼
                            Predicted Rating
                                0.5 – 5.0
        """,
        language="text"
    )

    st.subheader("📐 Architecture Details")

    architecture = [
        {
            "Component": "User Input",
            "Details": "User ID → Embedding",
            "Dimension": "32"
        },
        {
            "Component": "Movie Input",
            "Details": "Movie ID → Embedding",
            "Dimension": "32"
        },
        {
            "Component": "Poster Input",
            "Details": "CLIP embedding → Dense(32, ReLU)",
            "Dimension": "512 → 32"
        },
        {
            "Component": "Trailer Audio Input",
            "Details": "VGGish embedding → Dense(32, ReLU)",
            "Dimension": "128 → 32"
        },
        {
            "Component": "Fusion",
            "Details": "Concatenation",
            "Dimension": "128"
        },
        {
            "Component": "Hidden Layer 1",
            "Details": "Dense + ReLU",
            "Dimension": "64"
        },
        {
            "Component": "Regularization",
            "Details": "Dropout",
            "Dimension": "0.2"
        },
        {
            "Component": "Hidden Layer 2",
            "Details": "Dense + ReLU",
            "Dimension": "32"
        },
        {
            "Component": "Output",
            "Details": "Dense",
            "Dimension": "1 rating"
        }
    ]

    st.dataframe(
        architecture,
        use_container_width=True,
        hide_index=True
    )

    # DATA SOURCES
    st.header("🎞️ Multimodal Inputs")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("CLIP Poster", "512-D")
        st.write("Provides visual information from the movie poster.")

    with c2:
        st.metric("VGGish Audio", "128-D")
        st.write("Provides acoustic information from the movie trailer.")

    with c3:
        st.metric("MovieLens", "Ratings")
        st.write("Provides user–movie preference information.")

    st.info(
        "The project integrates MovieLens user preference data "
        "with CLIP poster embeddings and VGGish trailer-audio "
        "embeddings. The aligned multimodal catalog contains "
        "19,232 movies."
    )

    # RECOMMENDATION METHODS

    st.header("🎯 Recommendation Methods")

    st.markdown(
        """
        ### 👤 General Recommendations

        The trained multimodal model predicts ratings for unseen
        movies for a given user and ranks them from highest to lowest.

        ### 🎬 Movie-Based Recommendations

        A hybrid inference layer combines:

        **70% learned personalized rating**

        +

        **30% multimodal similarity to the selected movie**

        Poster/CLIP and trailer-audio embeddings are used for the
        movie similarity component.

        ### 🎭 Genre Recommendations

        The existing trained model is used without modification.
        Candidate movies are first filtered to the selected genre,
        and the model then ranks those movies according to the
        user's learned preferences.
        """
    )

    # COLD START

    st.header("❄️ Cold-Start Mechanism")

    st.markdown(
        """
        For movies with no training ratings, the system uses a
        dedicated cold-start multimodal model.

        An explicit **UNK movie index** represents the missing
        movie identity while the model still receives:

        - User representation
        - CLIP poster representation
        - Trailer-audio representation

        This allows the system to make predictions for movies
        with zero historical training ratings.
        """
    )

    # EVALUATION

    st.header("📊 Evaluation Results")

    results = [
        {
            "Model": "Global Mean",
            "Test MAE": 0.826679,
            "Test RMSE": 1.044473
        },
        {
            "Model": "User + Movie Bias",
            "Test MAE": 0.679932,
            "Test RMSE": 0.892608
        },
        {
            "Model": "NCF",
            "Test MAE": 0.825484,
            "Test RMSE": 1.043304
        },
        {
            "Model": "Multimodal",
            "Test MAE": 0.637302,
            "Test RMSE": 0.831881
        },
        {
            "Model": "Cold-Start Multimodal",
            "Test MAE": 0.673103,
            "Test RMSE": 0.894764
        }
    ]

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )
