import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Signals
N = 100
n = np.arange(N)

w1 = 0.2 * np.pi
w2 = 0.6 * np.pi

signal1 = np.sin(w1 * n)
signal2 = np.sin(w2 * n)

original_signal = signal1 + signal2

# 2. Add Gaussian Noise
noise = np.random.normal(0, 0.2, N)
noisy_signal = original_signal + noise

# 3. Frequency Range
omega = np.linspace(-np.pi, np.pi, 500)

# 4. Manual DTFT
def manual_dtft(signal, n, omega):

    X = np.zeros(len(omega), dtype=complex)

    for k in range(len(omega)):
        for i in range(len(n)):
            X[k] += signal[i] * np.exp(-1j * omega[k] * n[i])

    return X

# 5. Calculate DTFT
X_original = manual_dtft(original_signal, n, omega)
X_noisy = manual_dtft(noisy_signal, n, omega)

# 6. Magnitude
magnitude_original = np.abs(X_original)
magnitude_noisy = np.abs(X_noisy)

# 7. Phase
phase_original = np.angle(X_original)
phase_noisy = np.angle(X_noisy)

# 8. Plot Magnitude
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(omega, magnitude_original)
plt.title("Original Magnitude Spectrum")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(omega, magnitude_noisy)
plt.title("Noisy Magnitude Spectrum")
plt.grid()

plt.tight_layout()
plt.show()

# 9. Plot Phase
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(omega, phase_original)
plt.title("Original Phase Spectrum")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(omega, phase_noisy)
plt.title("Noisy Phase Spectrum")
plt.grid()

plt.tight_layout()
plt.show()