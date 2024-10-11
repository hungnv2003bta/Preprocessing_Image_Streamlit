# How to run: streamlit run ImageEnhancing.py

import streamlit as st
from PIL import Image
import cv2
import numpy as np

def create_noise(image, noise_type='gaussian', mean=0, var=0.1):
    src = np.array(image)
    
    if noise_type == 'gaussian':
        # Gaussian noise
        row, col, ch = src.shape
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma * 255, (row, col, ch))
        noisy = src + gauss
        noisy = np.clip(noisy, 0, 255)
        return noisy.astype(np.uint8)
    
    elif noise_type == 'salt_and_pepper':
        # Salt and pepper noise
        s_vs_p = 0.5
        amount = var
        out = np.copy(src)
        # Salt mode
        num_salt = np.ceil(amount * src.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in src.shape]
        out[coords[0], coords[1], :] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * src.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in src.shape]
        out[coords[0], coords[1], :] = 0
        return out
    
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

def blur_image(image, smooth):
    kernel = np.ones((smooth, smooth), np.float32) / (smooth ** 2)
    blurred = cv2.filter2D(np.array(image), -1, kernel)
    return blurred

def denoise_image(image, h=10, hForColor=10, templateWindowSize=7, searchWindowSize=21):
    # Apply Non-Local Means Denoising
    image_np = np.array(image)
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:  # If image is RGB
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        denoised_image = cv2.fastNlMeansDenoisingColored(image, None, h, hForColor, templateWindowSize, searchWindowSize)
        denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)
    else:
        denoised_image = cv2.fastNlMeansDenoising(np.array(image), None, h, templateWindowSize, searchWindowSize)
    return denoised_image

def sharpening_image(image, strength):
    image_np = np.array(image)
    kernel = np.array([[0, -1, 0], [-1, 4 + strength, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(src=image_np, ddepth = -1, kernel=kernel)
    return sharpened

def sobel_filter(image, strength=1.0):
    image_np = np.array(image)
    img_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0.1, 0.1)
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(img_gray, ddepth, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, ddepth, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    sobel_result = cv2.addWeighted(abs_grad_x, strength, abs_grad_y, strength, 0)
    return sobel_result

def prewitt_filter(image, strength=1.0):
    image_np = np.array(image)
    img_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0.1, 0.1)

    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    img_prewittx = cv2.filter2D(img_gray, cv2.CV_16S, kernelx)
    img_prewitty = cv2.filter2D(img_gray, cv2.CV_16S, kernely)

    abs_grad_x = cv2.convertScaleAbs(img_prewittx)
    abs_grad_y = cv2.convertScaleAbs(img_prewitty)

    prewitt_result = cv2.addWeighted(abs_grad_x, strength, abs_grad_y, strength, 0)
    return prewitt_result\
    
def canny_filter(image, strength=1.0):
    image_np = np.array(image)
    img_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0.1, 0.1)

    lower_threshold = int(50 * strength)
    upper_threshold = int(150 * strength)
    
    canny_filter = cv2.Canny(img_gray, lower_threshold, upper_threshold)
    return canny_filter

def main():
    st.title('Image Enhancing App')
    st.write('MSSV: 21522121 - NGUYEN VAN HUNG')
    st.markdown("<h1 style=' color: black;'>INPUT</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the uploaded image 
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)

        with st.sidebar:
            st.header("Image Enhancing Options")
            st.markdown("<h2 style='color: red;'>Adjust the settings to create noise</h2>", unsafe_allow_html=True)
            # add noise option
            noise_type = st.selectbox("Select Noise Type", ["gaussian", "salt_and_pepper"])
            var = st.slider("Noise Strength", 0.0, 1.0, 0.05)
            
            st.markdown("<h2 style='color: blue;'>Adjust the settings to enhance the image.</h2>", unsafe_allow_html=True)
            # Apply Blur option
            apply_blur = st.checkbox("Apply Blur")
            if apply_blur:
                smooth = st.slider("Blur Level", 0, 30, 5)
            else:
                smooth = 0

            # Apply Denoising option
            apply_denoise = st.checkbox("Apply Denoising (Non-Local Means)")
            if apply_denoise:
                h = st.slider("Denoising Strength", 0, 50, 10)

            # Apply Laplacian Filter with OpenCV
            apply_laplacian = st.checkbox("Apply Sharpening")
            if apply_laplacian:
                strength = st.slider("Strength", 0.5, 2.0, 1.1)
            
            st.markdown("<h2 style='color: green;'>Edge Detection Filter</h2>", unsafe_allow_html=True)
            # Apply Sobel Filter with OpenCV
            apply_sobel = st.checkbox("Apply Sobel Filter", key="sobel_checkbox")
            if apply_sobel:
                strength_sobel = st.slider("Strength", 0.0, 2.0, 1.1, key="sobel_strength")
            # Apply Prewitt Filter with OpenCV
            apply_prewitt = st.checkbox("Apply Prewitt Filter", key="prewitt_checkbox")
            if apply_prewitt:
                strength_prewitt = st.slider("Strength", 0.0, 2.0, 1.1, key="prewitt_strength")
            # Apply Canny Filter with OpenCV
            apply_canny = st.checkbox("Apply Canny Filter", key="canny_checkbox")
            if apply_canny: 
                strength_canny = st.slider("Strength", 0.0, 2.0, 1.1, key="canny_strength")

        enhanced_image = image 
        st.markdown("<h1 style=' color: red;'>Image after creating noise</h1>", unsafe_allow_html=True)
        enhanced_image = create_noise(enhanced_image, noise_type, var=var)
        st.image(enhanced_image, caption="Image after creating noise", width=300)
        
        st.markdown("<h1 style=' color: blue;'>Image after applying filter</h1>", unsafe_allow_html=True)
        if apply_blur and smooth > 0:
            enhanced_image = blur_image(enhanced_image, smooth)
        if apply_denoise:
            enhanced_image = denoise_image(enhanced_image, h=h)
        if apply_laplacian:
            enhanced_image = sharpening_image(enhanced_image, strength)
        # Display the output image only if any enhancement is applied
        if apply_blur or apply_denoise or apply_laplacian:
            st.image(enhanced_image, caption="Image after applying filter", width=300)

        st.markdown("<h1 style=' color: green;'>Edge Detection Filters From Input Image</h1>", unsafe_allow_html=True)
        if apply_sobel:
            sobel = sobel_filter(image, strength_sobel)
            st.image(sobel, caption="Sobel Filter", width=300)
        if apply_prewitt:
            prewitt = prewitt_filter(image, strength_prewitt)
            st.image(prewitt, caption="Prewitt Filter", width=300)
        if apply_canny:
            canny = canny_filter(image, strength_canny)
            st.image(canny, caption="Canny Filter", width=300)

if __name__ == "__main__":
    main()

