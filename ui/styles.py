"""Custom CSS styles for the Streamlit UI."""

import streamlit as st

CUSTOM_CSS = """
    <style>

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #111827,
            #312e81
        );
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        font-size: 2.5rem;
        margin-bottom: .4rem;
    }

    .movie-card {
        padding: 1.1rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 14px;
        min-height: 150px;
        margin-bottom: 1rem;
    }

    .score {
        font-size: 1.6rem;
        font-weight: 700;
    }

    .small {
        opacity: .75;
        font-size: .9rem;
    }

    .architecture-box {
        padding: 1rem;
        border-radius: 12px;
        background: rgba(128,128,128,.08);
        border: 1px solid rgba(128,128,128,.2);
    }

    </style>
    """


def apply_custom_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
