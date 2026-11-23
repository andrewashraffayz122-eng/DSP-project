import os
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="DSP Project", layout="wide")
st.title("Digital Signal Processing Project")

BASE = os.path.dirname(os.path.abspath(__file__))

# Marker where the image file is split into two tabs
SPLIT_MARKER = 'st.subheader("Notch Filtering")'

# (tab name, file name, part)  part: None = whole file, 0 = before marker, 1 = from marker
PAGES = [
    ("1. Signal & DTFT",           "dtft_streamlit.py", None),
    ("2. Convolution & Diff. Eq.", "convolution.py",    None),
    ("3. Audio: Add Noise",        "audio_noise.py",    None),
    ("4. Audio: Notch Filter",     "audio_notch.py",    None),
    ("5. Image: Filters & Noise",  "image_filter.py",   0),
    ("6. Image: Notch Removal",    "image_filter.py",   1),
]

# Shared namespaces so tab 6 can use the variables created in tab 5
SHARED = {}


def run_script(filename, part=None):
    path = os.path.join(BASE, filename)

    old_cfg, old_sidebar, old_show = st.set_page_config, st.sidebar, plt.show
    old_cwd = os.getcwd()

    try:
        st.set_page_config = lambda *a, **k: None
        st.sidebar = st.container()
        plt.show = lambda *a, **k: (st.pyplot(plt.gcf()), plt.close("all"))
        os.chdir(BASE)

        with open(path, encoding="utf-8") as f:
            source = f.read()

        if part is not None:
            if SPLIT_MARKER not in source:
                st.error(f"Split marker not found in {filename}")
                return
            before, after = source.split(SPLIT_MARKER, 1)
            source = before if part == 0 else SPLIT_MARKER + after

        # Same file => same namespace (shared between tab 5 and tab 6)
        ns = SHARED.setdefault(filename, {"__name__": "__main__", "__file__": path})
        exec(compile(source, path, "exec"), ns)

    except Exception as e:
        st.error(f"Error in {filename}: {e}")
        st.exception(e)
    finally:
        st.set_page_config, st.sidebar, plt.show = old_cfg, old_sidebar, old_show
        os.chdir(old_cwd)


tabs = st.tabs([name for name, _, _ in PAGES])
for tab, (_, filename, part) in zip(tabs, PAGES):
    with tab:
        run_script(filename, part)
