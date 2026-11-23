import io
import os 
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from scipy.signal import lfilter, freqz
from scipy.io import wavfile

# Run with: streamlit run app.py
st.set_page_config(page_title="DSP Project", layout="wide")
st.title("Digital Signal Processing Project")
st.caption("Signal Generation & DTFT • Convolution & Difference Equations • Image Processing • Audio Filtering")


# ------------------------- Shared helpers -------------------------
def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def normalize_audio(signal):
    signal = np.asarray(signal, dtype=float)
    peak = np.max(np.abs(signal)) if signal.size else 0.0
    if peak == 0:
        return signal
    return signal / peak


def wav_bytes(fs, signal):
    signal = np.asarray(signal, dtype=float)
    peak = np.max(np.abs(signal)) if signal.size else 0.0
    if peak > 0:
        signal = signal / peak
    pcm = np.int16(np.clip(signal, -1, 1) * 32767)
    buffer = io.BytesIO()
    wavfile.write(buffer, fs, pcm)
    buffer.seek(0)
    return buffer.getvalue()


def compute_spectrum(signal, fs):
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return np.array([]), np.array([])
    spectrum = np.abs(np.fft.rfft(signal)) / len(signal)
    frequencies = np.fft.rfftfreq(len(signal), d=1 / fs)
    return frequencies, spectrum


def manual_dtft(signal, n_values, omega_values):
    """Numerical DTFT using the defining summation."""
    result = np.zeros(len(omega_values), dtype=complex)
    for k, omega in enumerate(omega_values):
        result[k] = np.sum(signal * np.exp(-1j * omega * n_values))
    return result


def design_second_order_notch(f0, fs, quality_factor):
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * quality_factor)
    b = np.array([1, -2 * np.cos(w0), 1]) / (1 + alpha)
    a = np.array([
        1,
        -2 * np.cos(w0) / (1 + alpha),
        (1 - alpha) / (1 + alpha)
    ])
    return b, a


# ------------------------- Main application tabs -------------------------
tab_dtft, tab_conv, tab_image, tab_audio = st.tabs([
    "1. Signal & DTFT",
    "2. Convolution & Difference Equations",
    "3. Image Processing",
    "4. Audio Processing"
])


# ========================= 1. SIGNAL GENERATION & DTFT =========================
with tab_dtft:
    st.header("Signal Generation and DTFT")
    st.write("Generate two sinusoidal signals, combine them, add Gaussian noise, and compare their DTFT spectra.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        dtft_N = st.slider("Number of samples (N)", 50, 200, 100, key="dtft_N")
    with c2:
        dtft_w1 = st.slider("Frequency ω₁ (×π)", 0.01, 1.00, 0.20, key="dtft_w1") * np.pi
    with c3:
        dtft_w2 = st.slider("Frequency ω₂ (×π)", 0.01, 1.00, 0.60, key="dtft_w2")
        dtft_w2 *= np.pi
    with c4:
        dtft_noise_std = st.slider("Noise standard deviation", 0.0, 1.0, 0.2, key="dtft_noise")

    dtft_n = np.arange(dtft_N)
    signal1 = np.sin(dtft_w1 * dtft_n)
    signal2 = np.sin(dtft_w2 * dtft_n)
    original_signal = signal1 + signal2
    rng = np.random.default_rng(42)
    gaussian_noise = rng.normal(0, dtft_noise_std, dtft_N)
    noisy_signal = original_signal + gaussian_noise

    omega = np.linspace(-np.pi, np.pi, 500, endpoint=False)
    X_original = manual_dtft(original_signal, dtft_n, omega)
    X_noisy = manual_dtft(noisy_signal, dtft_n, omega)

    st.subheader("Time-domain signals")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dtft_n, signal1, label="Signal 1")
    ax.plot(dtft_n, signal2, label="Signal 2")
    ax.plot(dtft_n, original_signal, label="Combined signal", linewidth=2)
    ax.set_xlabel("n")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend()
    show_fig(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(dtft_n, noisy_signal, label="Noisy signal")
    ax.plot(dtft_n, original_signal, alpha=0.7, label="Original signal")
    ax.set_xlabel("n")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend()
    show_fig(fig)

    st.subheader("DTFT Magnitude and Phase")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(omega / np.pi, np.abs(X_original), label="Original")
        ax.plot(omega / np.pi, np.abs(X_noisy), label="Noisy", alpha=0.8)
        ax.set_xlabel("Normalized frequency (ω/π)")
        ax.set_ylabel("|X(ω)|")
        ax.set_title("Magnitude Spectrum")
        ax.grid(True)
        ax.legend()
        show_fig(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(omega / np.pi, np.angle(X_original), label="Original")
        ax.plot(omega / np.pi, np.angle(X_noisy), label="Noisy", alpha=0.8)
        ax.set_xlabel("Normalized frequency (ω/π)")
        ax.set_ylabel("Phase (radians)")
        ax.set_title("Phase Spectrum")
        ax.grid(True)
        ax.legend()
        show_fig(fig)

    with st.expander("Manual DTFT equation"):
        st.latex(r"X(\omega)=\sum_{n=-\infty}^{\infty}x[n]e^{-j\omega n}")
        st.write("The code evaluates this summation numerically for each frequency in the selected ω range.")


# ========================= 2. CONVOLUTION & DIFFERENCE EQUATIONS =========================
with tab_conv:
    st.header("Convolution and Difference Equations")
    st.write("Generate a noisy sinusoidal input, then examine moving-average convolution and a first-order difference equation.")

    c1, c2, c3 = st.columns(3)
    with c1:
        conv_N = st.slider("Number of samples", 20, 500, 100, key="conv_N")
        conv_freq = st.slider("Input frequency", 0.01, 0.45, 0.05, key="conv_freq")
    with c2:
        conv_noise = st.slider("Noise amplitude", 0.0, 2.0, 0.5, key="conv_noise")
        conv_M = st.slider("Moving average length (M)", 1, 30, 5, key="conv_M")
    with c3:
        conv_a = st.slider("Difference equation coefficient (a)", -1.2, 1.2, 0.8, key="conv_a")
        conv_seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42, key="conv_seed")

    conv_n = np.arange(conv_N)
    conv_rng = np.random.default_rng(int(conv_seed))
    conv_x = np.sin(2 * np.pi * conv_freq * conv_n) + conv_noise * conv_rng.standard_normal(conv_N)
    conv_h = np.ones(conv_M) / conv_M
    conv_y = np.convolve(conv_x, conv_h, mode="full")

    st.subheader("Moving Average Convolution")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(conv_n, conv_x)
    ax.set_title("Input signal x[n]")
    ax.set_xlabel("n")
    ax.grid(True)
    show_fig(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(np.arange(len(conv_y)), conv_y)
    ax.set_title("Linear convolution y[n] = x[n] * h[n]")
    ax.set_xlabel("n")
    ax.grid(True)
    show_fig(fig)

    # FFT implementation of linear convolution (zero-padding to N + M - 1)
    fft_length = conv_N + conv_M - 1
    conv_y_fft = np.fft.ifft(
        np.fft.fft(conv_x, fft_length) * np.fft.fft(conv_h, fft_length)
    ).real
    max_error = float(np.max(np.abs(conv_y - conv_y_fft)))
    st.metric("Maximum difference: direct vs FFT convolution", f"{max_error:.3e}")
    st.caption("Zero-padding to N + M − 1 produces the same linear convolution using the FFT.")

    st.subheader("Difference Equation")
    st.latex(r"y[n]=a\,y[n-1]+x[n]")
    diff_y = np.zeros(conv_N)
    for i in range(conv_N):
        diff_y[i] = conv_x[i] + (conv_a * diff_y[i - 1] if i > 0 else 0)

    diff_y_check = lfilter([1], [1, -conv_a], conv_x)
    st.write("Implementation check against scipy.signal.lfilter:",
             "Passed" if np.allclose(diff_y, diff_y_check) else "Not matched")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(conv_n, conv_x, label="Input x[n]", alpha=0.7)
    ax.plot(conv_n, diff_y, label="Output y[n]")
    ax.set_title("Difference Equation Input and Output")
    ax.set_xlabel("n")
    ax.grid(True)
    ax.legend()
    show_fig(fig)

    if abs(conv_a) < 1:
        st.success("The system is BIBO stable because |a| < 1.")
    else:
        st.warning("The system is not BIBO stable because |a| ≥ 1.")

    impulse = np.zeros(conv_N)
    impulse[0] = 1
    impulse_response = lfilter([1], [1, -conv_a], impulse)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.stem(conv_n, impulse_response)
    ax.set_title("Impulse Response")
    ax.set_xlabel("n")
    ax.grid(True)
    show_fig(fig)

    st.subheader("Moving Average Frequency Response")
    H = np.fft.fft(conv_h, 1024)
    f = np.fft.fftfreq(1024, d=1)
    positive = f >= 0
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(f[positive], np.abs(H[positive]))
    ax.axvline(conv_freq, linestyle="--", label="Input frequency")
    ax.set_xlabel("Frequency (cycles/sample)")
    ax.set_ylabel("|H(f)|")
    ax.grid(True)
    ax.legend()
    show_fig(fig)


# ========================= 3. IMAGE PROCESSING =========================

with tab_image:
    st.header("Image Processing using 2D Fourier Transform")

    # Load image directly from project folder
    image_path = os.path.join(os.path.dirname(__file__), "image.jpg")

    if not os.path.exists(image_path):
        st.error("Image file not found. Put image.jpg inside the project folder.")

    else:
        image = Image.open(image_path).convert("L")
        image_array = np.asarray(image, dtype=float)

        st.subheader("Original and Grayscale Image")

        c1, c2 = st.columns(2)

        with c1:
            st.image(
                Image.open(image_path),
                caption="Original image",
                use_container_width=True
            )

        with c2:
            st.image(
                image_array.astype(np.uint8),
                caption="Grayscale image",
                use_container_width=True
            )

        rows, cols = image_array.shape

        st.write(f"Image dimensions: **{cols} × {rows}**")

        # 2D Fourier Transform
        F = np.fft.fftshift(np.fft.fft2(image_array))
        magnitude = np.log1p(np.abs(F))

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.imshow(magnitude, cmap="gray")
        ax.set_title("Centered 2D Fourier Magnitude Spectrum")
        ax.axis("off")
        show_fig(fig)

        # Ideal Low-Pass and High-Pass Filters
        st.subheader("Ideal Low-Pass and High-Pass Filters")

        c1, c2 = st.columns(2)

        with c1:
            lp_radius = st.slider(
                "Low-pass radius",
                5,
                max(6, min(rows, cols) // 2),
                min(40, max(5, min(rows, cols) // 4)),
                key="lp_radius"
            )

        with c2:
            hp_radius = st.slider(
                "High-pass radius",
                5,
                max(6, min(rows, cols) // 2),
                min(20, max(5, min(rows, cols) // 8)),
                key="hp_radius"
            )

        center_y, center_x = rows // 2, cols // 2

        yy, xx = np.ogrid[:rows, :cols]

        distance = np.sqrt(
            (yy - center_y) ** 2 +
            (xx - center_x) ** 2
        )

        low_mask = (distance <= lp_radius).astype(float)
        high_mask = (distance >= hp_radius).astype(float)

        low_result = np.abs(
            np.fft.ifft2(
                np.fft.ifftshift(F * low_mask)
            )
        )

        high_result = np.abs(
            np.fft.ifft2(
                np.fft.ifftshift(F * high_mask)
            )
        )

        # Function to display images correctly
        def display_image(array):
            maximum = np.max(array) if array.size else 0

            if maximum > 0:
                array = array / maximum * 255

            return np.clip(array, 0, 255).astype(np.uint8)

        c1, c2 = st.columns(2)

        with c1:
            st.image(
                display_image(low_result),
                caption="Ideal Low-Pass Filtered",
                use_container_width=True
            )

        with c2:
            st.image(
                display_image(high_result),
                caption="Ideal High-Pass Filtered",
                use_container_width=True
            )

        # Periodic Noise and Notch Filtering
        st.subheader("Periodic Noise and Notch Filtering")

        c1, c2, c3 = st.columns(3)

        with c1:
            noise_fx = st.slider(
                "Noise frequency X",
                1,
                max(2, cols // 4),
                min(20, max(1, cols // 10)),
                key="noise_fx"
            )

        with c2:
            noise_fy = st.slider(
                "Noise frequency Y",
                1,
                max(2, rows // 4),
                min(20, max(1, rows // 10)),
                key="noise_fy"
            )

        with c3:
            noise_amp = st.slider(
                "Noise amplitude",
                0,
                100,
                40,
                key="image_noise_amp"
            )

        y_grid, x_grid = np.mgrid[0:rows, 0:cols]

        periodic_noise = noise_amp * np.sin(
            2 * np.pi * (
                noise_fx * x_grid / cols +
                noise_fy * y_grid / rows
            )
        )

        noisy_image = image_array + periodic_noise

        F_noisy = np.fft.fftshift(
            np.fft.fft2(noisy_image)
        )

        c1, c2 = st.columns(2)

        with c1:
            st.image(
                display_image(noisy_image),
                caption="Image with periodic noise",
                use_container_width=True
            )

        with c2:
            # Normalize Fourier spectrum to [0, 255]
            noisy_spectrum = np.log1p(np.abs(F_noisy))

            max_value = noisy_spectrum.max()

            if max_value > 0:
                noisy_spectrum = (
                    noisy_spectrum / max_value * 255
                )

            noisy_spectrum = noisy_spectrum.astype(np.uint8)

            st.image(
                noisy_spectrum,
                caption="Noisy image Fourier spectrum",
                use_container_width=True
            )

        # Notch Filter
        notch_radius = st.slider(
            "Notch radius",
            1,
            30,
            5,
            key="notch_radius"
        )

        notch_mask = np.ones((rows, cols), dtype=float)

        for py, px in [
            (center_y + noise_fy, center_x + noise_fx),
            (center_y - noise_fy, center_x - noise_fx)
        ]:
            notch_distance = np.sqrt(
                (yy - py) ** 2 +
                (xx - px) ** 2
            )

            notch_mask[notch_distance <= notch_radius] = 0

        restored_image = np.abs(
            np.fft.ifft2(
                np.fft.ifftshift(F_noisy * notch_mask)
            )
        )

        # Compare Original, Noisy and Restored Images
        c1, c2, c3 = st.columns(3)

        with c1:
            st.image(
                display_image(image_array),
                caption="Original",
                use_container_width=True
            )

        with c2:
            st.image(
                display_image(noisy_image),
                caption="Noisy",
                use_container_width=True
            )

        with c3:
            st.image(
                display_image(restored_image),
                caption="Restored after notch filter",
                use_container_width=True
            )

        # Calculate MSE and PSNR
        mse = float(
            np.mean((image_array - restored_image) ** 2)
        )

        if mse == 0:
            psnr = float("inf")
        else:
            psnr = float(
                10 * np.log10((255 ** 2) / mse)
            )

        c1, c2 = st.columns(2)

        c1.metric(
            "Mean Squared Error (MSE)",
            f"{mse:.3f}"
        )

        c2.metric(
            "Peak Signal-to-Noise Ratio (PSNR)",
            "∞ dB" if np.isinf(psnr) else f"{psnr:.2f} dB"
        )


# ========================= 4. AUDIO PROCESSING =========================
with tab_audio:
    st.header("Audio Processing: Interference Noise and Notch Filter")
    st.write("Upload a clean WAV file, add 50/60 Hz interference, then remove it using a second-order IIR notch filter.")

    audio_file = st.file_uploader("Upload clean audio (WAV)", type=["wav"], key="audio_upload")

    if audio_file is None:
        st.info("Upload a WAV audio file to continue.")
    else:
        try:
            fs, audio_raw = wavfile.read(audio_file)
            if audio_raw.ndim > 1:
                audio_raw = audio_raw[:, 0]
            audio_clean = normalize_audio(audio_raw)
            if audio_clean.size == 0:
                st.error("The uploaded audio file is empty.")
            elif np.max(np.abs(audio_clean)) == 0:
                st.error("The uploaded audio is silent. Please upload a non-silent WAV file.")
            else:
                st.success(f"Audio loaded successfully — sampling frequency: {fs} Hz")
                st.audio(wav_bytes(fs, audio_clean), format="audio/wav")

                st.subheader("Step 1: Add Interference Noise")
                c1, c2 = st.columns(2)
                with c1:
                    mains_frequency = st.selectbox("Interference frequency", [50, 60], index=0, key="audio_noise_frequency")
                with c2:
                    noise_amplitude = st.slider("Noise amplitude", 0.05, 1.0, 0.3, key="audio_noise_amplitude")

                if mains_frequency >= fs / 2:
                    st.error(f"{mains_frequency} Hz is not below the Nyquist frequency ({fs/2:.1f} Hz). Choose a WAV file with a higher sampling rate.")
                else:
                    audio_t = np.arange(len(audio_clean)) / fs
                    interference = noise_amplitude * np.sin(2 * np.pi * mains_frequency * audio_t)
                    audio_noisy = audio_clean + interference

                    st.subheader("Step 2: Inspect the Noisy Signal")
                    c1, c2 = st.columns(2)
                    with c1:
                        fig, ax = plt.subplots(figsize=(8, 3))
                        time_limit = min(len(audio_t), max(1, int(fs * 0.05)))
                        ax.plot(audio_t[:time_limit], audio_clean[:time_limit], label="Clean")
                        ax.plot(audio_t[:time_limit], audio_noisy[:time_limit], label="Noisy", alpha=0.8)
                        ax.set_xlabel("Time (seconds)")
                        ax.set_title("Time Domain")
                        ax.grid(True)
                        ax.legend()
                        show_fig(fig)
                    with c2:
                        f_clean, s_clean = compute_spectrum(audio_clean, fs)
                        f_noisy, s_noisy = compute_spectrum(audio_noisy, fs)
                        fig, ax = plt.subplots(figsize=(8, 3))
                        ax.plot(f_clean, s_clean, label="Clean")
                        ax.plot(f_noisy, s_noisy, label="Noisy", alpha=0.8)
                        ax.set_xlim(0, min(500, fs / 2))
                        ax.set_xlabel("Frequency (Hz)")
                        ax.set_title("Magnitude Spectrum")
                        ax.grid(True)
                        ax.legend()
                        show_fig(fig)

                    st.audio(wav_bytes(fs, audio_noisy), format="audio/wav")
                    st.download_button(
                        "Download noisy audio",
                        data=wav_bytes(fs, audio_noisy),
                        file_name="noisy_audio.wav",
                        mime="audio/wav"
                    )

                    st.subheader("Step 3: Second-Order Notch Filter")
                    detected_f0 = float(mains_frequency)
                    f0_choice = st.selectbox(
                        "Notch frequency",
                        [detected_f0, 50, 60],
                        index=0,
                        format_func=lambda value: f"{int(value)} Hz" + (" (selected interference)" if value == detected_f0 else ""),
                        key="notch_frequency"
                    )
                    Q = st.slider("Quality factor (Q)", 5.0, 60.0, 30.0, key="notch_Q")

                    if f0_choice >= fs / 2:
                        st.error("The selected notch frequency must be below the Nyquist frequency.")
                    else:
                        b, a = design_second_order_notch(f0_choice, fs, Q)
                        audio_filtered = lfilter(b, a, audio_noisy)
                        filtered_freq, filtered_spectrum = compute_spectrum(audio_filtered, fs)

                        w, h = freqz(b, a, worN=2048, fs=fs)
                        pole_radius = float(np.max(np.abs(np.roots(a))))
                        stable = pole_radius < 1

                        st.latex(
                            r"H(z)=\frac{1-2\cos(\omega_0)z^{-1}+z^{-2}}"
                            r"{1-2r\cos(\omega_0)z^{-1}+r^2z^{-2}}"
                        )
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Notch frequency", f"{f0_choice:.0f} Hz")
                        c2.metric("Quality factor Q", f"{Q:.1f}")
                        c3.metric("Maximum pole radius", f"{pole_radius:.5f}")
                        if stable:
                            st.success("The IIR notch filter is stable (all poles are inside the unit circle).")
                        else:
                            st.warning("The filter is not stable.")

                        st.write("Numerator coefficients (b):", b)
                        st.write("Denominator coefficients (a):", a)

                        # Estimate attenuation using the nearest FFT bin to the notch frequency.
                        if len(f_noisy) > 1:
                            bin_index = int(np.argmin(np.abs(f_noisy - f0_choice)))
                            before_level = float(s_noisy[bin_index])
                            after_level = float(filtered_spectrum[bin_index])
                            attenuation_db = 20 * np.log10(
                                max(before_level, 1e-12) / max(after_level, 1e-12)
                            )
                            st.metric("Attenuation near notch frequency", f"{attenuation_db:.2f} dB")
                        else:
                            st.info("Not enough samples to estimate attenuation.")

                        st.subheader("Filter Frequency Response")
                        fig, ax = plt.subplots(figsize=(9, 3.5))
                        ax.plot(w, 20 * np.log10(np.maximum(np.abs(h), 1e-12)))
                        ax.axvline(f0_choice, color="red", linestyle="--", label=f"Notch at {f0_choice:.0f} Hz")
                        ax.set_xlim(0, min(500, fs / 2))
                        ax.set_ylim(-100, 5)
                        ax.set_xlabel("Frequency (Hz)")
                        ax.set_ylabel("Magnitude (dB)")
                        ax.grid(True)
                        ax.legend()
                        show_fig(fig)

                        st.subheader("Before and After Filtering")
                        c1, c2 = st.columns(2)
                        with c1:
                            fig, ax = plt.subplots(figsize=(8, 3))
                            ax.plot(audio_t[:time_limit], audio_noisy[:time_limit], label="Noisy")
                            ax.plot(audio_t[:time_limit], audio_filtered[:time_limit], label="Filtered", alpha=0.8)
                            ax.set_xlabel("Time (seconds)")
                            ax.set_title("Time-Domain Comparison")
                            ax.grid(True)
                            ax.legend()
                            show_fig(fig)
                        with c2:
                            fig, ax = plt.subplots(figsize=(8, 3))
                            ax.plot(f_noisy, s_noisy, label="Before filtering")
                            ax.plot(filtered_freq, filtered_spectrum, label="After filtering")
                            ax.set_xlim(0, min(500, fs / 2))
                            ax.set_xlabel("Frequency (Hz)")
                            ax.set_title("Spectrum Comparison")
                            ax.grid(True)
                            ax.legend()
                            show_fig(fig)

                        st.audio(wav_bytes(fs, audio_filtered), format="audio/wav")
                        st.download_button(
                            "Download filtered audio",
                            data=wav_bytes(fs, audio_filtered),
                            file_name="filtered_audio.wav",
                            mime="audio/wav"
                        )

                        with st.expander("Theoretical explanation"):
                            st.markdown(
                                """
                                - A **notch filter** rejects a narrow band of frequencies around the selected interference frequency.
                                - The **Q factor** controls the notch width: a larger Q gives a narrower notch.
                                - This is an **IIR filter**, because its output depends on previous output samples.
                                - The filter is stable when all denominator roots (poles) lie inside the unit circle.
                                """
                            )
        except Exception as error:
            st.error(f"Could not read or process this WAV file: {error}")