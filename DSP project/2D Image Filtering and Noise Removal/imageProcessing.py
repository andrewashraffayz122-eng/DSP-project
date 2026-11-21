import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(
    page_title= "Image Filtering and 2D Fourier Transform",
    layout= 'wide'
)

st.title("Image Filtering and 2D Fourier Transform")


st.divider()

image= Image.open("image.jpg") #change png to jpg in case the loading time causes issues

greyImage= image.convert("L")
imageArray= np.array(greyImage)

st.header("Original vs. Greyscale Images")

col1, col2, col3= st.columns([2,1,2])

with col1:
    st.subheader("Original Image")
    st.image(image, use_container_width= True)

with col2:
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    convert= st.button('Convert to Greyscale', use_container_width= True)

with col3:
    st.subheader("Greyscale Version")
    if convert:
        st.image(greyImage, use_container_width= True)
    else:
        st.info("Click the button to show the converted image.")

st.divider()

st.header("2D Fourier Transform")

imageArray= np.array(greyImage)
fourierTransform= np.fft.fft2(imageArray)
fourierShifted= np.fft.fftshift(fourierTransform)
magnitudeSpectrum= np.abs(fourierShifted)
spectrumDisplay= np.log1p(magnitudeSpectrum)

col1, col2= st.columns(2)

with col1:
    st.subheader("Greyscale Image")
    st.image(greyImage, use_container_width= True)

with col2:
    st.subheader('2D Fourier Magnitude Spectrum')
    fig, ax= plt.subplots(figsize=(6, 4))

    ax.imshow(spectrumDisplay, cmap= "grey")
    ax.set_xlabel("Horizontal Spatial Frequency")
    ax.set_ylabel("Vertical Spatial Frequency")
    ax.set_title("2D Fourier Transform Spectrum")
    fig.tight_layout(pad= 1.0)

    st.pyplot(fig, use_container_width= True)
    plt.close(fig)

st.subheader("Fourier Transform Computation")
st.write("Original Image Dimensions: ", imageArray.shape)
st.write("Fourier Transform Dimensions: ", fourierTransform.shape)
st.write("Magnitude Spectrum Dimensions: ", magnitudeSpectrum.shape)

st.divider()

st.header("Ideal Low-Pass and Ideal High-Pass Image Filtering")

st.subheader("Greyscale Version of the Image")
st.image(greyImage, use_container_width= True)

st.divider()

st.subheader("Comparison between Lowpass and Highpass Image Filtering")

rows, cols= imageArray.shape
cRow, cCol= rows//2, cols//2

y, x= np.ogrid[:rows, :cols]
distance= np.sqrt((x-cCol)**2 + (y-cRow)**2)

fourierShifted= np.fft.fftshift(np.fft.fft2(imageArray))

col1, col2= st.columns(2)

with col1:
    st.subheader("Lowpass Image Filtering")
    lpfRadius= st.slider(
        "Lowpass Radius",
        min_value= 0,
        max_value= min(rows, cols)//2,
        value= 50,
        key= "lpf"
    )

    lowPassMask= (distance <= lpfRadius)
    lowPassResult= fourierShifted* lowPassMask

    lowPassImg= np.fft.ifft2(
        np.fft.ifftshift(lowPassResult)
    )

    lowPassImg= np.abs(lowPassImg)
    lowPassImg = (lowPassImg / lowPassImg.max() * 255).astype(np.uint8)
    st.image(lowPassImg, use_container_width= True, clamp= True)


with col2:
    st.subheader('Highpass Image Filtering')
    hpfRadius= st.slider(
        "Highpass Radius",
        min_value= 0,
        max_value= min(rows, cols)//2,
        value= 50,
        key= "hpf"
    )

    highPassMask= (distance >= hpfRadius)
    highPassResult= fourierShifted* highPassMask

    highPassImg= np.fft.ifft2(
        np.fft.ifftshift(highPassResult)
    )

    highPassImg= np.abs(highPassImg)
    highPassImg = (highPassImg / highPassImg.max() * 255).astype(np.uint8)
    st.image(highPassImg, use_container_width= True, clamp= True)


st.divider()

#=========================Image Noise Removal & Evaluation

st.header("Periodic Noise Removal and Image Restoration")
st.subheader("Adding Periodic Sinusoidal Noise")

rows, cols = imageArray.shape
y, x = np.indices((rows, cols))
col1, col2 = st.columns(2)

with col1:
    noiseFreqX = st.slider(
        "Horizontal Noise Frequency",
        min_value=1,
        max_value=max(1, cols // 4),
        value=10,
        key="noise_x"
    )

with col2:
    noiseFreqY = st.slider(
        "Vertical Noise Frequency",
        min_value=1,
        max_value=max(1, rows // 4),
        value=10,
        key="noise_y"
    )

noiseAmplitude = 40

noise = noiseAmplitude * np.sin(
    2 * np.pi * (
        noiseFreqX * x / cols +
        noiseFreqY * y / rows
    )
)
noisyImageFloat = imageArray.astype(float) + noise
noisyImage = np.clip(noisyImageFloat, 0, 255).astype(np.uint8)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Grayscale Image")
    st.image(greyImage, use_container_width=True)

with col2:
    st.subheader("Image with Periodic Noise")
    st.image(noisyImage, use_container_width=True)


st.divider()

st.subheader("Fourier Spectrum of Noisy Image")
noisyFourier = np.fft.fft2(noisyImageFloat)
noisyFourierShifted = np.fft.fftshift(noisyFourier)
noisyMagnitude = np.abs(noisyFourierShifted)
noisySpectrum = np.log1p(noisyMagnitude)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Noisy Grayscale Image")
    st.image(noisyImage, use_container_width=True)

with col2:
    st.subheader("Noisy Image Fourier Spectrum")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(noisySpectrum, cmap="gray")
    ax.set_title("Fourier Spectrum with Periodic Noise")
    ax.set_xlabel("Horizontal Frequency")
    ax.set_ylabel("Vertical Frequency")
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


st.divider()

st.subheader("Notch Filtering")
cRow, cCol = rows // 2, cols // 2
peak1Row = cRow + noiseFreqY
peak1Col = cCol + noiseFreqX

peak2Row = cRow - noiseFreqY
peak2Col = cCol - noiseFreqX

notchRadius = st.slider(
    "Notch Filter Radius",
    min_value=1,
    max_value=30,
    value=5,
    key="notch_radius"
)

Y, X = np.ogrid[:rows, :cols]
distance1 = np.sqrt(
    (Y - peak1Row)**2 + (X - peak1Col)**2
)

distance2 = np.sqrt(
    (Y - peak2Row)**2 + (X - peak2Col)**2
)


notchMask = (
    (distance1 > notchRadius) &
    (distance2 > notchRadius)
)

filteredFourier = noisyFourierShifted * notchMask
restoredImageFloat = np.fft.ifft2(
    np.fft.ifftshift(filteredFourier)
)

restoredImageFloat = np.real(restoredImageFloat)
restoredImage = np.clip(
    restoredImageFloat, 0, 255
).astype(np.uint8)


col1, col2 = st.columns(2)

with col1:
    st.subheader("Notch Mask")
    st.image(
        notchMask.astype(np.uint8) * 255,
        use_container_width=True
    )

with col2:
    st.subheader("Restored Image")
    st.image(restoredImage, use_container_width=True)

st.divider()

st.header("Image Restoration Comparison")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Original Image")
    st.image(greyImage, use_container_width=True)

with col2:
    st.subheader("Noisy Image")
    st.image(noisyImage, use_container_width=True)

with col3:
    st.subheader("Restored Image")
    st.image(restoredImage, use_container_width=True)


originalFloat = imageArray.astype(float)

mseNoisy = np.mean(
    (originalFloat - noisyImageFloat)**2
)

mseRestored = np.mean(
    (originalFloat - restoredImageFloat)**2
)



def calculatePSNR(mse):
    if mse == 0:
        return float("inf")

    return 10 * np.log10((255**2) / mse)


psnrNoisy = calculatePSNR(mseNoisy)
psnrRestored = calculatePSNR(mseRestored)

st.divider()

st.subheader("Restoration Quality Evaluation")

col1, col2 = st.columns(2)

with col1:
    st.metric("Noisy Image MSE", f"{mseNoisy:.2f}")
    st.metric("Noisy Image PSNR", f"{psnrNoisy:.2f} dB")

with col2:
    st.metric("Restored Image MSE", f"{mseRestored:.2f}")
    st.metric("Restored Image PSNR", f"{psnrRestored:.2f} dB")
