import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Signal Processing Application")
st.write("Signal Generation and DTFT Analysis")

# 1. User Inputs
st.sidebar.header("Signal Settings")

N = st.sidebar.slider("Number of Samples", 50, 200, 100)

w1 = st.sidebar.slider(
    "First Frequency (× π)", 0.05, 0.95, 0.2
) * np.pi

w2 = st.sidebar.slider(
    "Second Frequency (× π)", 0.05, 0.95, 0.6
) * np.pi

noise_std = st.sidebar.slider(
    "Noise Strength", 0.0, 1.0, 0.2
)

# 2. Generate Signals
n = np.arange(N)

signal1 = np.sin(w1 * n)
signal2 = np.sin(w2 * n)

original_signal = signal1 + signal2

# 3. Add Noise
noise = np.random.normal(0, noise_std, N)
noisy_signal = original_signal + noise

# 4. Frequency Range
omega = np.linspace(-np.pi, np.pi, 500, endpoint=False)

# 5. Manual DTFT
def manual_dtft(signal, n, omega):
    X = np.zeros(len(omega), dtype=complex)

    for k in range(len(omega)):
        for i in range(len(n)):
            X[k] += signal[i] * np.exp(
                -1j * omega[k] * n[i]
            )

    return X

# 6. Calculate DTFT
X_original = manual_dtft(original_signal, n, omega)
X_noisy = manual_dtft(noisy_signal, n, omega)

# 7. Magnitude and Phase
magnitude_original = np.abs(X_original)
magnitude_noisy = np.abs(X_noisy)

phase_original = np.angle(X_original)
phase_noisy = np.angle(X_noisy)

# 8. Plot Signals
st.header("Time Domain Signals")

fig1, ax = plt.subplots(figsize=(10, 4))
ax.plot(n, original_signal, label="Original Signal")
ax.plot(n, noisy_signal, label="Noisy Signal", alpha=0.7)
ax.set_xlabel("n")
ax.set_ylabel("Amplitude")
ax.legend()
ax.grid()

st.pyplot(fig1)

# 9. Magnitude Spectrum
st.header("Magnitude Spectrum")

fig2, ax = plt.subplots(figsize=(10, 4))
ax.plot(omega / np.pi, magnitude_original,
        label="Original")
ax.plot(omega / np.pi, magnitude_noisy,
        label="Noisy", alpha=0.7)
ax.set_xlabel("Frequency (× π)")
ax.set_ylabel("Magnitude")
ax.legend()
ax.grid()

st.pyplot(fig2)

# 10. Phase Spectrum
st.header("Phase Spectrum")

fig3, ax = plt.subplots(figsize=(10, 4))
ax.plot(omega / np.pi, phase_original,
        label="Original")
ax.plot(omega / np.pi, phase_noisy,
        label="Noisy", alpha=0.7)
ax.set_xlabel("Frequency (× π)")
ax.set_ylabel("Phase (Radians)")
ax.legend()
ax.grid()

st.pyplot(fig3)