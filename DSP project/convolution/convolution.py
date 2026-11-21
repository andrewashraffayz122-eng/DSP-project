import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.signal import lfilter

st.set_page_config(page_title="DSP - Student 2", layout="wide")
st.title("Student 2: Convolution & Difference Equations")

# ---------------- Sidebar controls ----------------
st.sidebar.header("Parameters")
N = st.sidebar.slider("N (signal length)", 20, 500, 100)
freq = st.sidebar.slider("Sine frequency (cycles/sample)", 0.01, 0.45, 0.05, 0.01)
noise = st.sidebar.slider("Noise strength", 0.0, 2.0, 0.5, 0.1)
M = st.sidebar.slider("M (Moving Average length)", 1, 30, 5)
a = st.sidebar.slider("a (difference equation)", -1.2, 1.2, 0.8, 0.05)
pad = st.sidebar.checkbox("Use zero-padding (L = N + M - 1)", True)
seed = st.sidebar.number_input("Random seed", 0, 9999, 0)

# ---------------- Signal ----------------
np.random.seed(int(seed))
n = np.arange(N)
x = np.sin(2 * np.pi * freq * n) + noise * np.random.randn(N)

tab1, tab2, tab3 = st.tabs(
    ["Moving Average: Time vs Frequency", "Difference Equation", "Frequency Response"]
)

# ---------------- Tab 1: Moving average ----------------
with tab1:
    h = np.ones(M) / M
    y_conv = np.convolve(x, h)                      # length N + M - 1
    L = N + M - 1 if pad else N
    y_freq = np.fft.ifft(np.fft.fft(x, L) * np.fft.fft(h, L)).real

    k = min(len(y_conv), len(y_freq))
    same = np.allclose(y_conv[:k], y_freq[:k])
    c1, c2, c3 = st.columns(3)
    c1.metric("len(np.convolve)", len(y_conv))
    c2.metric("L used in FFT", L)
    c3.metric("Results match?", "Yes" if same else "No")
    if not pad:
        st.info("Without zero-padding the FFT computes a CIRCULAR convolution, "
                "so the first M-1 samples are corrupted.")

    fig, ax = plt.subplots(3, 1, figsize=(9, 7))
    ax[0].stem(x); ax[0].set_title("Input x[n] (noisy sine)")
    ax[1].stem(y_conv); ax[1].set_title(f"Moving Average via np.convolve (M={M})")
    ax[2].stem(y_freq); ax[2].set_title(f"Frequency-domain method (L={L})")
    plt.tight_layout()
    st.pyplot(fig)

# ---------------- Tab 2: Difference equation ----------------
with tab2:
    st.latex(r"y[n] = a\,y[n-1] + x[n]")
    y_diff = np.zeros(N)
    for i in range(N):
        prev = y_diff[i - 1] if i > 0 else 0.0
        y_diff[i] = a * prev + x[i]

    ok = np.allclose(y_diff, lfilter([1], [1, -a], x))
    st.write("Loop matches scipy `lfilter`:", ok)
    if abs(a) >= 1:
        st.warning("|a| >= 1: the system is NOT stable, the output grows.")
    else:
        st.success("|a| < 1: the system is stable.")

    imp = np.zeros(30); imp[0] = 1
    h_diff = lfilter([1], [1, -a], imp)

    fig, ax = plt.subplots(2, 1, figsize=(9, 6))
    ax[0].stem(y_diff); ax[0].set_title(f"Output y[n] (a = {a})")
    ax[1].stem(h_diff); ax[1].set_title("Impulse response h[n] = a^n u[n]")
    plt.tight_layout()
    st.pyplot(fig)

# ---------------- Tab 3: Frequency response ----------------
with tab3:
    st.write("Magnitude of H(w) for the Moving Average filter: it is a Low-Pass.")
    h = np.ones(M) / M
    H = np.fft.rfft(h, 1024)
    w = np.linspace(0, 0.5, len(H))   # cycles/sample
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(w, np.abs(H))
    ax.axvline(freq, color="r", linestyle="--", label=f"sine freq = {freq}")
    ax.set_xlabel("Frequency (cycles/sample)")
    ax.set_ylabel("|H|")
    ax.legend(); ax.grid(True)
    st.pyplot(fig)
