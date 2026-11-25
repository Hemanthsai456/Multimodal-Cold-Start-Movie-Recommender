"""Page 1 — Recommendations UI."""

import sys
from pathlib import Path
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def render_recommendations(rec):
    st.markdown(
        """
        <div class="hero">
        <h1>Multimodal Movie Recommender</h1>
        <p>
        Generate recommendations using user preferences,
        movie similarity, or genre preferences.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # PROJECT STATS
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Users",
        f"{rec.stats['users']:,}"
    )

    c2.metric(
        "Multimodal Movies",
        f"{rec.stats['multimodal_movies']:,}"
    )

    c3.metric(
        "Embedding Dimensions",
        f"CLIP {rec.stats['clip_dim']} · "
        f"Audio {rec.stats['audio_dim']}"
    )

    st.divider()

    # RECOMMENDATION MODE

    st.subheader("🎯 Recommendation Mode")
    mode = st.radio(
        "Choose how you want recommendations",
        [
            "👤 General Recommendations",
            "🎬 Based on a Movie",
            "🎭 By Genre"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.divider()

    # MODE 1 — GENERAL
    if mode == "👤 General Recommendations":

        st.subheader("👤 General Personalized Recommendations")

        st.write(
            "The trained model ranks unseen movies according "
            "to the user's learned preferences."
        )

        user_id = st.number_input(
            "MovieLens User ID",
            min_value=1,
            value=1,
            step=1,
            key="general_user"
        )

        top_n = st.slider(
            "Number of recommendations",
            min_value=5,
            max_value=20,
            value=10,
            key="general_top_n"
        )

        if st.button(
            "✨ Generate Recommendations",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Generating personalized recommendations..."):
                results = rec.recommend_movies(
                    user_id=int(user_id),
                    top_n=top_n
                )

            if results.empty:
                st.warning(
                    "No recommendations found. "
                    "Check the User ID."
                )

            else:
                st.success(
                    f"Recommendations generated for "
                    f"User {int(user_id)}."
                )

                st.subheader("🍿 Recommended For You")

                cols = st.columns(2)
                for i, row in results.iterrows():
                    with cols[i % 2]:
                        st.markdown(
                            f"""
                            <div class="movie-card">

                            <h3>
                            #{int(row['Rank'])} · {row['Movie']}
                            </h3>

                            <p>
                            {str(row['Genres']).replace('|', ' · ')}
                            </p>

                            <div class="score">
                            ⭐ {row['Predicted Rating']:.2f} / 5
                            </div>

                            <p class="small">
                            {row['Model Type']}
                            ·
                            {row['Training Status']}
                            </p>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    # MODE 2 — MOVIE BASED

    elif mode == "🎬 Based on a Movie":

        st.subheader("🎬 Recommendations Based on a Movie")

        st.write(
            "Choose a movie you like. The system combines "
            "the user's learned preferences with multimodal "
            "similarity to the selected movie."
        )

        user_id = st.number_input(
            "MovieLens User ID",
            min_value=1,
            value=1,
            step=1,
            key="movie_user"
        )

        query = st.text_input(
            "🔎 Search for a movie you like",
            placeholder=(
                "Try: Interstellar, "
                "Dark Knight, Toy Story..."
            ),
            key="movie_search"
        )

        selected_movie_id = None
        movie_options = {}

        if query.strip():
            found = rec.search_movies(
                query.strip(),
                top_n=20
            )

            if found.empty:
                st.info(
                    "No matching movies found. "
                    "Try another title."
                )

            else:
                movie_options = {
                    int(row["movieId"]):
                    (
                        f"{row['title']} — "
                        f"{str(row['genres']).replace('|', ' · ')}"
                    )
                    for _, row in found.iterrows()
                }

                selected_movie_id = st.selectbox(
                    "Select the movie",
                    list(movie_options.keys()),
                    format_func=lambda x:
                        movie_options[x],
                    key="selected_movie"
                )

        top_n = st.slider(
            "Number of recommendations",
            min_value=5,
            max_value=20,
            value=10,
            key="movie_top_n"
        )

        if st.button(
            "✨ Recommend Based on Movie",
            type="primary",
            use_container_width=True,
            disabled=selected_movie_id is None
        ):

            with st.spinner("Combining user taste with movie similarity..."):
                results = rec.recommend_from_movie(
                    user_id=int(user_id),
                    movie_id=int(selected_movie_id),
                    top_n=top_n
                )

            if results.empty:
                st.warning(
                    "No recommendations found. "
                    "Check the User ID or selected movie."
                )

            else:
                selected_title = (
                    movie_options[selected_movie_id]
                    .split(" — ")[0]
                )

                st.success(
                    f"Recommendations generated using "
                    f"User {int(user_id)}'s taste and "
                    f"{selected_title}."
                )

                st.subheader("🍿 Recommended For You")

                cols = st.columns(2)

                for i, row in results.iterrows():
                    with cols[i % 2]:
                        st.markdown(
                            f"""
                            <div class="movie-card">

                            <h3>
                            #{int(row['Rank'])} · {row['Movie']}
                            </h3>

                            <p>
                            {str(row['Genres']).replace('|', ' · ')}
                            </p>

                            <div class="score">
                            ⭐ {row['Predicted Rating']:.2f} / 5
                            </div>

                            <p class="small">
                            Personalized:
                            {row['personalized_rating']:.2f}
                            ·
                            Movie similarity:
                            {row['Similarity Score']:.2f}
                            </p>

                            <p class="small">
                            {row['Model Type']}
                            ·
                            {row['Training Status']}
                            </p>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

        st.caption(
            "Hybrid ranking: 70% learned personalized rating "
            "+ 30% multimodal similarity to the selected movie "
            "(poster/CLIP + trailer audio)."
        )

    # MODE 3 — GENRE

    else:

        st.subheader("🎭 Personalized Recommendations by Genre")

        st.write(
            "Choose a genre and the trained model will rank "
            "movies from that genre according to the user's "
            "learned preferences."
        )

        # USER
        user_id = st.number_input(
            "MovieLens User ID",
            min_value=1,
            value=1,
            step=1,
            key="genre_user"
        )

        # GENRE LIST

        genres = [
            "All Genres",
            "Action",
            "Adventure",
            "Animation",
            "Children",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "IMAX",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western"
        ]

        selected_genre = st.selectbox(
            "🎭 Choose a genre",
            genres,
            key="selected_genre"
        )

        top_n = st.slider(
            "Number of recommendations",
            min_value=5,
            max_value=20,
            value=10,
            key="genre_top_n"
        )

        if st.button(
            "✨ Recommend in This Genre",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Finding movies matching the user's taste..."):

                if selected_genre == "All Genres":
                    results = rec.recommend_movies(
                        user_id=int(user_id),
                        top_n=top_n
                    )

                else:
                    results = rec.recommend_movies(
                        user_id=int(user_id),
                        top_n=top_n,
                        genre=selected_genre
                    )

            if results.empty:
                st.warning(
                    "No recommendations found for this "
                    "user and genre."
                )

            else:
                if selected_genre == "All Genres":
                    st.success(
                        f"Personalized recommendations "
                        f"generated for User {int(user_id)}."
                    )

                else:
                    st.success(
                        f"Personalized {selected_genre} "
                        f"recommendations generated for "
                        f"User {int(user_id)}."
                    )

                st.subheader("🍿 Recommended Movies")

                cols = st.columns(2)

                for i, row in results.iterrows():
                    with cols[i % 2]:

                        st.markdown(
                            f"""
                            <div class="movie-card">

                            <h3>
                            #{int(row['Rank'])} · {row['Movie']}
                            </h3>

                            <p>
                            {str(row['Genres']).replace('|', ' · ')}
                            </p>

                            <div class="score">
                            ⭐ {row['Predicted Rating']:.2f} / 5
                            </div>

                            <p class="small">
                            {row['Model Type']}
                            ·
                            {row['Training Status']}
                            </p>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )
