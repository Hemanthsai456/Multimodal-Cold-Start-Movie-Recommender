import sys
from pathlib import Path
import streamlit as st

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from recommender import MovieRecommender
from ui.about import render_about
from ui.cold_start import render_cold_start
from ui.recommendations import render_recommendations
from ui.styles import apply_custom_css

# PAGE CONFIG

st.set_page_config(
    page_title="Multimodal Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# CUSTOM CSS

apply_custom_css()

# LOAD MODEL

ARTIFACTS_DIR = _ROOT / "recommended_artifacts"


@st.cache_resource(show_spinner=False)
def load_recommender():
    return MovieRecommender(ARTIFACTS_DIR)


try:
    rec = load_recommender()
except Exception as e:
    st.error("Unable to load recommendation model.")
    st.code(str(e))
    st.stop()

# SIDEBAR

page = st.sidebar.radio(
    "Pages",
    [
        "🎬 Recommendations",
        "❄️ Cold Start",
        "🧠 About Model"
    ]
)

if page == "🎬 Recommendations":
    render_recommendations(rec)
elif page == "❄️ Cold Start":
    render_cold_start(rec)
elif page == "🧠 About Model":
    render_about()
