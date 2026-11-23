
import io
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.io import wavfile

st.title("Audio Analysis - Interference Tone")

uploaded = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded is not None:

    fs, x = wavfile.read(uploaded)

    if x.ndim > 1:
        x = x[:, 0]

    x = x.astype(float)
    x = x / np.max(np.abs(x))

    f_noise = st.radio("Noise frequency (Hz)", [50, 60])
    amp = st.slider("Amplitude", 0.05, 1.0, 0.3)

    N = len(x)                    
    t = np.arange(N) / fs          

    noise = amp * np.sin(2 * np.pi * f_noise * t)
    x_noisy = x + noise


    def spectrum(sig):

        mag = np.abs(np.fft.rfft(sig)) / len(sig)
        freqs = np.fft.rfftfreq(len(sig), d=1 / fs)
        return freqs, mag

    f_clean, mag_clean = spectrum(x)
    f_noisy, mag_noisy = spectrum(x_noisy)

    mask = (f_noisy > 20) & (f_noisy < 100)        
    idx = np.argmax(mag_noisy[mask])                 
    detected = f_noisy[mask][idx]                   

    st.success(f"Detected noise frequency ≈ {detected:.1f} Hz")
    st.write(f"Sampling frequency = {fs} Hz  |  Nyquist = {fs / 2} Hz")

    n_show = min(N, fs)
    fig1, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].plot(t[:n_show], x[:n_show])
    ax[0].set_title("Clean signal (Time domain)")
    ax[1].plot(t[:n_show], x_noisy[:n_show], color="red")
    ax[1].set_title("Noisy signal (Time domain)")
    ax[1].set_xlabel("Time (s)")
    plt.tight_layout()
    st.pyplot(fig1)


    fig2, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].plot(f_clean, mag_clean)
    ax[0].set_xlim(0, 500)
    ax[0].set_title("Clean spectrum")
    ax[1].plot(f_noisy, mag_noisy, color="red")
    ax[1].set_xlim(0, 500)
    ax[1].set_title("Noisy spectrum (notice the peak)")
    ax[1].axvline(detected, color="k", linestyle="--", label=f"{detected:.1f} Hz")
    ax[1].legend()
    ax[1].set_xlabel("Frequency (Hz)")
    plt.tight_layout()
    st.pyplot(fig2)


    out = x_noisy / np.max(np.abs(x_noisy))
    out_int16 = (out * 32767).astype(np.int16)

    buf = io.BytesIO()
    wavfile.write(buf, fs, out_int16)

    st.subheader("Listen")
    st.write("Clean WAV file:")
    st.audio(uploaded)
    st.write("With noise:")
    st.audio(buf.getvalue(), format="audio/wav")

    st.download_button("Download noisy.wav", buf.getvalue(),
                       file_name="noisy.wav", mime="audio/wav")
