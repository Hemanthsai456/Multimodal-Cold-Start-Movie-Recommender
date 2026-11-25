"""Page 2 — Cold Start UI."""

import sys
from pathlib import Path
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def render_cold_start(rec):
    st.markdown(
        """
        <div class="hero">

        <h1>Cold-Start Recommendation</h1>

        <p>
        Demonstration of the cold-start model for movies
        without historical training interactions.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    user_id = st.number_input(
        "MovieLens User ID",
        min_value=1,
        value=1,
        step=1,
        key="cold_user"
    )

    cold = rec.cold_start_movies(50)

    if cold.empty:

        st.error(
            "No unseen movies were found in the saved catalog."
        )

    else:

        labels = {
            int(r.movieId):
            (
                f"{r.title} — "
                f"{str(r.genres).replace('|', ' · ')}"
            )
            for _, r in cold.iterrows()
        }

        selected = st.selectbox(
            "Choose a catalog movie with no training interaction",
            list(labels),
            format_func=lambda x: labels[x]
        )

        if st.button(
            "❄️ Predict Cold-Start Rating",
            type="primary",
            use_container_width=True
        ):

            result = rec.predict_movie_rating(user_id, selected)

            if result["status"] == "success":
                st.success(
                    f"Predicted rating: "
                    f"{result['predicted_rating']:.2f} / 5"
                )

                a, b = st.columns(2)
                a.metric("Movie", result["title"])
                b.metric("Model", result["model_type"])

                st.write(
                    f"**Genres:** "
                    f"{str(result['genres']).replace('|', ' · ')}"
                )

                st.write(
                    f"**Training status:** "
                    f"{result['training_status']}"
                )

            else:
                st.error(result["message"])
