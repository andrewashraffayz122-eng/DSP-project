import os
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="DSP Project", layout="wide")
st.title("Digital Signal Processing Project")

BASE = os.path.dirname(os.path.abspath(__file__))

PAGES = [
    ("1. Signal & DTFT",           "dtft_streamlit.py"),
    ("2. Convolution & Diff. Eq.", "convolution.py"),
    ("3. Audio: Add Noise",        "audio_noise.py"),
    ("4. Audio: Notch Filter",     "audio_notch.py"),
    ("5. Image Processing",        "image_filter.py"),
]


def run_script(filename):
    """Run an untouched script inside the current tab."""
    path = os.path.join(BASE, filename)

    old_cfg, old_sidebar, old_show = st.set_page_config, st.sidebar, plt.show
    old_cwd = os.getcwd()

    try:
        st.set_page_config = lambda *a, **k: None
        st.sidebar = st.container()
        plt.show = lambda *a, **k: (st.pyplot(plt.gcf()), plt.close("all"))
        os.chdir(BASE)

        with open(path, encoding="utf-8") as f:
            code = compile(f.read(), path, "exec")
        exec(code, {"__name__": "__main__", "__file__": path})

    except Exception as e:
        st.error(f"Error in {filename}: {e}")
        st.exception(e)
    finally:
        st.set_page_config, st.sidebar, plt.show = old_cfg, old_sidebar, old_show
        os.chdir(old_cwd)


tabs = st.tabs([name for name, _ in PAGES])
for tab, (_, filename) in zip(tabs, PAGES):
    with tab:
        run_script(filename)
