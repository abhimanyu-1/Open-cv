import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
import io
from matplotlib import pyplot as plt
import os 
import math
from ImgPreprocessing.process import *
import tempfile
from ImgPreprocessing.code_generator import generate_function_code

##################################################  STREAMLIT ####################################################

st.set_page_config(
    page_title="Image Preprocessing Suite",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)
#st.markdown( "",unsafe_allow_html=True)
#st.markdown('<h1 class="main-header" style="text-align:center; padding-bottom:0px">Image Preprocessing</h1>', unsafe_allow_html=True)

# Initialize session state variables
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'cumulative_image' not in st.session_state:
    st.session_state.cumulative_image = None
if 'output' not in st.session_state:
    st.session_state.output = ""
if 'temp_path' not in st.session_state:
    st.session_state.temp_path = ""

# Convert PIL image into cv2
def pil_to_cv2(pil_image):
    open_cv_image = np.array(pil_image)
    if len(open_cv_image.shape) == 3:
        open_cv_image = cv.cvtColor(open_cv_image, cv.COLOR_RGB2BGR)
    return open_cv_image

# Convert cv2 image into PIL
def cv2_to_pil(cv_image):
    if len(cv_image.shape) == 3:
        cv_image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
    return Image.fromarray(cv_image)

with st.sidebar:
    st.header("🔧 Controls")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to process"
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.session_state.original_image = pil_to_cv2(image)

            if st.session_state.original_image is not None:
                if st.session_state.cumulative_image is None:
                    st.session_state.cumulative_image = st.session_state.original_image.copy()

            st.success("Image uploaded successfully!")

            # Save temp file path to session state
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                cv.imwrite(temp_file.name, st.session_state.original_image)
                st.session_state.temp_path = temp_file.name.replace("\\", "/")

            #if st.button("Reset Image"):
            #    st.session_state.cumulative_image = st.session_state.original_image.copy()
            #    st.session_state.output = ""
            #    st.rerun()
                
        except Exception as e:
            st.error(f"Error in File uploading: {e}")

    if st.session_state.original_image is not None:
        st.divider()
        
        operation = st.selectbox(
            "🎯 Select Operation",
            [
                "Original Image",
                "Global Threshold",
                "Adaptive Threshold",
                "Otsu's Threshold",
                "Box Detection",
                "Skew Correction",
                "Watermark Remove",
                "Line Detection",
                "Horizontal Line Detection", 
                "Vertical Line Detection",
                "Horizontal Line Removal",
                "Vertical Line Removal",
                "Dotted Line Detection",
                "Mean Blur",
                "Gaussian Blur",
                "Median Blur",
                "Bilateral Filtering",
                "Erosion",
                "Dilation",
                "Opening",
                "Closing",
                "Noise Removal",
                "Face Detection", 
                "Strike Removal",
                "Image Inpainting"
            ]
        )
        
        st.divider()
        
        # Parameters based on operation
        if operation == "Global Threshold":
            thresh_type = st.selectbox("Select Type",
                [
                    "Binary",
                    "Binary Inverted",
                    "Truncate",
                    "To Zero",
                    "To Zero Inverted"
                ]
            )
            thresh = st.slider("Threshold", 1, 255, 125, step=5)
            maxVal = st.slider("Maximum Value", 1, 255, 255, step=5)

        elif operation == "Adaptive Threshold":
            op_type = st.selectbox("Select Type",
                [
                    "Mean",
                    "Gaussian",
                ]
            )
            thresh_type = st.selectbox("Select Type",
                [
                    "Binary",
                    "Binary Inverted",
                ]
            )
            maxVal = st.slider("Maximum Value", 1, 255, 125, step=5)
            blockSize = st.slider("Block Size", 1, 100, 3, step=2)
            C = st.slider("C", 1, 100, 1, step=2)
        
        elif operation == "Otsu's Threshold":
            op_type = st.selectbox("Select Type",
                [
                    "Binary",
                    "Binary Inverted",
                ]
            )
            maxVal = st.slider("Maximum Value", 1, 255, 255, step=5)
        
        elif operation == "Box Detection":
            mode = st.selectbox("Select mode",
                [
                    "External",
                    "List",
                    "CCOMP",
                    "Tree"
                ]
            )
            method = st.selectbox("Select method",
                [
                    "None",
                    "Simple"
                ]
            )
            width = st.slider("Minimum Width", 1, 100, 40, step=5)
            height = st.slider("Minimum height", 1, 100, 40, step=5)

        elif operation == "Skew Correction":
            line_angle = st.slider("Line Angle", 1, 360, 90)
            min_line_len = st.slider("Minimum Line Length", 1, 300, 100)
            max_line_gap = st.slider("Maximum Line Gap", 1, 100, 5)

        elif operation == "Watermark Remove":
            morph_radius = st.slider("Morph Radius", 1, 10, 4)
            dilation_iter = st.slider("Dilation Iterations", 1, 10, 1)
        
        elif operation == "Image Inpainting":
            mask_img = None
            mask_path = None
            mask = st.file_uploader("Upload the mask image", type=["png","jpg","jpeg"])
            if mask is not None:
                try:
                    mask_img = Image.open(mask)
                    tmp_img = pil_to_cv2(mask_img)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                        cv.imwrite(temp_file.name, tmp_img)
                        mask_path = temp_file.name.replace("\\", "/")
                    st.image(mask_img, caption="Mask", width=150)
                except Exception as e:
                    st.warning(e)
            inpaint_method = st.selectbox("Select method",
                [
                    "TELEA",
                    "NS"
                ]
            )
            inpaint_radius = st.slider("Inpaint Radius", 1, 30, 3)
        
        elif operation in ["Horizontal Line Removal", "Vertical Line Removal"]:
            constant = st.slider("Constant", 1, 30, 1)
            kernel = st.slider("Kernel", 1, 30, 1)  # Fixed typo: keranl -> kernel
            iteration = st.slider("Iteration", 1, 10, 1)
            neighbor = st.slider("Neighbor", 11, 200, step=4)
        
        elif operation in ["Line Detection", "Horizontal Line Detection", "Vertical Line Detection"]:
            threshold = st.slider("Threshold", 1, 200, 30, help="Hough transform threshold")
            min_length = st.slider("Min Line Length", 1, 100, 40, help="Minimum line length")
            max_gap = st.slider("Max Line Gap", 1, 50, 10, help="Maximum gap between line segments")

        elif operation == "Dotted Line Detection":
            st.subheader("📏 Dotted Line Parameters")
            min_len = st.slider("Min Length", 1, 50, 10, help="Minimum line length")
            max_len = st.slider("Max Length", 1, 30, 5, help="Maximum line length")
            row_tolerance = st.slider("Row Tolerance", 1, 20, 10, help="Row tolerance for horizontal lines")
            col_tolerance = st.slider("Column Tolerance", 1, 20, 10, help="Column tolerance for vertical lines")
            
        elif operation in ["Mean Blur", "Gaussian Blur", "Median Blur"]:
            st.subheader("Blur Parameters")
            ksize = st.slider("Kernel Size", 1, 31, 5, step=2, help="Blur kernel size (odd numbers only)")
            if operation == "Gaussian Blur":
                sigma_x = st.slider("Sigma X", 0.0, 10.0, 1.0, help="Gaussian kernel standard deviation")
                
        elif operation == "Bilateral Filtering":
            st.subheader("Bilateral Filter Parameters")
            d = st.slider("Diameter", 1, 20, 9, help="Diameter of pixel neighborhood")
            sigma_color = st.slider("Sigma Color", 1, 200, 75, help="Filter sigma in color space")
            sigma_space = st.slider("Sigma Space", 1, 200, 75, help="Filter sigma in coordinate space")
            
        elif operation in ["Erosion", "Dilation", "Opening", "Closing"]:
            st.subheader("Morphological Parameters")
            iterations = st.slider("Iterations", 1, 10, 1, help="Number of iterations")
            kernel_size = st.slider("Kernel Size", 3, 15, 5, step=2, help="Morphological kernel size")

        elif operation == "Noise Removal":
            st.subheader("Noise Removal")
            h = st.slider("Luminance", 1, 50, 1, help="Higher value more smoother")
            hColor = st.slider("Strength", 10, 50, 2, help="Strength of filter")
            templateWindowSize = st.slider("window size", 1, 20, 1, help="Compare similarity")
            searchWindowSize = st.slider("Search window", 1, 50, 1, help="Find patches")
        
        elif operation == "Face Detection":
            st.subheader("Face Detection")
            scale = st.slider("Scale", min_value=1.05, max_value=1.5, value=1.1, step=0.01)
            neighbor = st.slider("neighbor", min_value=1, max_value=10, value=5, step=1)
        
        elif operation == "Strike Removal":
            st.subheader("Strike Remover")
            paint = st.slider("paint", 3, 30, 2)
            neighbor = st.slider("Neighbor", 11, 200, step=4)
            kernel = st.slider("Kernel", 1, 30, 1)
            iteration = st.slider("Iteration", 1, 10, 1)
        
        # Apply Changes button
        col1, col2 = st.columns([2, 1])  # Two equally sized columns

        with col1:
            save_pressed = st.button("Apply Changes", type="primary")

        with col2:
            if st.button("Reset Image"):
                st.session_state.cumulative_image = st.session_state.original_image.copy()
                st.session_state.output = ""
                st.rerun()

# Main processing logic
if st.session_state.original_image is not None:
    if "cumulative_image" not in st.session_state:
        st.session_state.cumulative_image = st.session_state.original_image.copy()
    
    # Fixed: Added missing comma
    cumulative_operations = [
        "Horizontal Line Removal", "Vertical Line Removal", 
        "Mean Blur", "Gaussian Blur", "Median Blur", "Bilateral Filtering",
        "Erosion", "Dilation", "Opening", "Closing", "Global Threshold", "Adaptive Threshold",  # Fixed comma
        "Skew Correction", "Watermark Remove", "Image Inpainting", "Noise Removal", "Face Detection", "Strike Removal"
    ]
    
    detection_operations = [
        "Line Detection", "Horizontal Line Detection", "Vertical Line Detection", "Dotted Line Detection", "Box Detection"
    ]
    
    try:
        if operation == "Original Image":
            st.session_state.processed_image = st.session_state.original_image.copy()

        elif operation == "Global Threshold":
            st.session_state.processed_image = global_threshold(
                st.session_state.original_image, thresh_type, thresh, maxVal
            )
            if save_pressed:
                st.session_state.cumulative_image = global_threshold(
                    st.session_state.cumulative_image, thresh_type, thresh, maxVal
                )

        elif operation == "Adaptive Threshold":
            if op_type == "Mean":
                st.session_state.processed_image = adaptive_mean_threshold(
                    st.session_state.original_image, maxVal, thresh_type, blockSize, C
                )
                if save_pressed:
                    st.session_state.cumulative_image = adaptive_mean_threshold(
                        st.session_state.cumulative_image, maxVal, thresh_type, blockSize, C
                    )
            elif op_type == "Gaussian":
                st.session_state.processed_image = adaptive_gaussian_threshold(
                    st.session_state.original_image, maxVal, thresh_type, blockSize, C
                )
                if save_pressed:
                    st.session_state.cumulative_image = adaptive_gaussian_threshold(
                        st.session_state.cumulative_image, maxVal, thresh_type, blockSize, C
                    )

        elif operation == "Otsu's Threshold":
            st.session_state.processed_image = otsu_threshold(
                st.session_state.original_image, op_type, maxVal
            )
            if save_pressed:
                st.session_state.cumulative_image = otsu_threshold(
                    st.session_state.cumulative_image, op_type, maxVal
                )

        elif operation == "Box Detection":
            st.session_state.processed_image = box_detection(
                st.session_state.original_image, width, height, mode, method
            )

        elif operation == "Skew Correction":
            st.session_state.processed_image = skew_houghLine(
                st.session_state.original_image, line_angle, min_line_len, max_line_gap
            )
            if save_pressed:
                st.session_state.cumulative_image = skew_houghLine(
                    st.session_state.cumulative_image, line_angle, min_line_len, max_line_gap
                )

        elif operation == "Watermark Remove":
            st.session_state.processed_image = watermark_remove(
                st.session_state.original_image, morph_radius, dilation_iter
            )
            if save_pressed:
                st.session_state.cumulative_image = watermark_remove(
                    st.session_state.cumulative_image, morph_radius, dilation_iter
                )

        elif operation == "Image Inpainting":
            if mask_img is not None:
                st.session_state.processed_image = ImageInpainting(
                    st.session_state.original_image, mask_img, inpaint_method, inpaint_radius
                )
                if save_pressed:
                    st.session_state.cumulative_image = ImageInpainting(
                        st.session_state.cumulative_image, mask_img, inpaint_method, inpaint_radius
                    )

        elif operation == "Line Detection":
            st.session_state.processed_image = line_detection(
                st.session_state.original_image, threshold, min_length, max_gap
            )

        elif operation == "Horizontal Line Detection":
            st.session_state.processed_image = horizontal_detection(
                st.session_state.original_image, threshold, min_length, max_gap
            )

        elif operation == "Vertical Line Detection":
            st.session_state.processed_image = vertical_detection(
                st.session_state.original_image, threshold, min_length, max_gap
            )

        elif operation == "Horizontal Line Removal":
            st.session_state.processed_image = horizontal_line_removal(
                st.session_state.original_image, constant, neighbor, kernel, iteration  # Fixed: keranl -> kernel
            )
            if save_pressed:
                st.session_state.cumulative_image = horizontal_line_removal(
                    st.session_state.cumulative_image, constant, neighbor, kernel, iteration
                )

        elif operation == "Vertical Line Removal":
            st.session_state.processed_image = vertical_line_removal(
                st.session_state.original_image, constant, neighbor, kernel, iteration  # Fixed: keranl -> kernel
            )
            if save_pressed:
                st.session_state.cumulative_image = vertical_line_removal(
                    st.session_state.cumulative_image, constant, neighbor, kernel, iteration
                )

        elif operation == "Dotted Line Detection":
            st.session_state.processed_image = dotted_line_detection(
                st.session_state.original_image, min_len, max_len, row_tolerance, col_tolerance
            )

        elif operation == "Mean Blur":
            st.session_state.processed_image = mean_blur(
                st.session_state.original_image, ksize
            )
            if save_pressed:
                st.session_state.cumulative_image = mean_blur(
                    st.session_state.cumulative_image, ksize
                )

        elif operation == "Gaussian Blur":
            st.session_state.processed_image = gaussian_blur(
                st.session_state.original_image, ksize, sigma_x
            )
            if save_pressed:
                st.session_state.cumulative_image = gaussian_blur(
                    st.session_state.cumulative_image, ksize, sigma_x
                )

        elif operation == "Median Blur":
            st.session_state.processed_image = median_blur(
                st.session_state.original_image, ksize
            )
            if save_pressed:
                st.session_state.cumulative_image = median_blur(
                    st.session_state.cumulative_image, ksize
                )

        elif operation == "Bilateral Filtering":
            st.session_state.processed_image = bilateral_filtering(
                st.session_state.original_image, d, sigma_color, sigma_space
            )
            if save_pressed:
                st.session_state.cumulative_image = bilateral_filtering(
                    st.session_state.cumulative_image, d, sigma_color, sigma_space
                )

        elif operation == "Erosion":
            st.session_state.processed_image = erosion(
                st.session_state.original_image, iterations, kernel_size
            )
            if save_pressed:
                st.session_state.cumulative_image = erosion(
                    st.session_state.cumulative_image, iterations, kernel_size
                )

        elif operation == "Dilation":
            st.session_state.processed_image = dilation(
                st.session_state.original_image, iterations, kernel_size
            )
            if save_pressed:
                st.session_state.cumulative_image = dilation(
                    st.session_state.cumulative_image, iterations, kernel_size
                )

        elif operation == "Opening":
            st.session_state.processed_image = opening(
                st.session_state.original_image, iterations, kernel_size
            )
            if save_pressed:
                st.session_state.cumulative_image = opening(
                    st.session_state.cumulative_image, iterations, kernel_size
                )

        elif operation == "Closing":
            st.session_state.processed_image = closing(
                st.session_state.original_image, iterations, kernel_size
            )
            if save_pressed:
                st.session_state.cumulative_image = closing(
                    st.session_state.cumulative_image, iterations, kernel_size
                )

        elif operation == "Noise Removal":
            st.session_state.processed_image = NoiseRemoval(
                st.session_state.original_image, h, hColor, templateWindowSize, searchWindowSize
            )
            if save_pressed:
                st.session_state.cumulative_image = NoiseRemoval(
                    st.session_state.cumulative_image, h, hColor, templateWindowSize, searchWindowSize
                )
        
        elif operation == "Face Detection":
            st.session_state.processed_image = face_detect(
                st.session_state.original_image, scale, neighbor
            )
            if save_pressed:
                st.session_state.cumulative_image = face_detect(
                    st.session_state.cumulative_image, scale, neighbor
                )
    
        elif operation == 'Strike Removal':
            st.session_state.processed_image = LineOverText(
                st.session_state.original_image, paint, neighbor, kernel, iteration
            )
            if save_pressed:
                st.session_state.cumulative_image = LineOverText(
                    st.session_state.cumulative_image, paint, neighbor, kernel, iteration
                )
            
    except Exception as e:
        st.error(f"An exception occurred while preprocessing '{operation}': {str(e)}")
        st.stop()

    # Display images
    col1, col2 = st.columns(2)

    try:
        with col1:
            st.markdown("<h6 style='text-aligh:center;'> Original Image </h6>" , unsafe_allow_html=True)
            st.image(cv2_to_pil(st.session_state.original_image), width=450)

        with col2:
            #st.subheader(f"{operation}")
            st.markdown(f"<h6 style='text-aligh:center;'> {operation} </h6>", unsafe_allow_html=True)
            if st.session_state.processed_image is not None:
                st.image(cv2_to_pil(st.session_state.processed_image), width=450)

        # Add some vertical spacing
        st.markdown("<br>", unsafe_allow_html=True)

        # Center the bottom image using 3 columns
        left, center, right = st.columns([1, 2, 1])
        with center:
            st.markdown("<h2 style='text-aligh:center; margin-left:100px; '> Final Result </h2>" , unsafe_allow_html=True)
            st.image(cv2_to_pil(st.session_state.cumulative_image), width=400)

    except Exception as e:
        st.error(f"An error occurred while displaying image: {e}")
        st.stop()



    # Download buttons - Images and Code
    col1, col2 , col3 = st.columns([5,4,10])
    
    try:
        with col1:
            pil_cumulative = cv2_to_pil(st.session_state.cumulative_image)
            img_buffer_cum = io.BytesIO()
            pil_cumulative.save(img_buffer_cum, format='PNG')
            img_buffer_cum.seek(0)

        with col2:
            # Initialize code if empty
            if len(st.session_state.output) == 0:
                st.session_state.output = f"import cv2 as cv\nimport numpy as np\n\nimg = cv.imread('{st.session_state.temp_path}')\n"

            # Generate code when save is pressed
            if save_pressed:
                params = {}
                if operation == "Global Threshold":
                    params = {"thresh_type": thresh_type, "thresh": thresh, "maxVal": maxVal}
                elif operation == "Adaptive Threshold":
                    params = {"op_type": op_type, "maxVal": maxVal, "thresh_type": thresh_type, "blockSize": blockSize, "C": C}
                elif operation == "Otsu's Threshold":
                    params = {"op_type": op_type, "maxVal": maxVal}
                elif operation == "Box Detection":
                    params = {"width": width, "height": height, "mode": mode, "method": method}
                elif operation == "Skew Correction":
                    params = {"line_angle": line_angle, "min_line_len": min_line_len, "max_line_gap": max_line_gap}
                elif operation == "Watermark Remove":
                    params = {"morph_radius": morph_radius, "dilation_iter": dilation_iter}
                elif operation in ["Horizontal Line Removal", "Vertical Line Removal"]:
                    params = {"constant": constant, "kernel": kernel, "iteration": iteration, "neighborhood": neighbor}
                elif operation in ["Line Detection", "Horizontal Line Detection", "Vertical Line Detection"]:
                    params = {"threshold": threshold, "min_length": min_length, "max_gap": max_gap}
                elif operation == "Dotted Line Detection":
                    params = {"min_len": min_len, "max_len": max_len, "row_tolerance": row_tolerance, "col_tolerance": col_tolerance}
                elif operation in ["Mean Blur", "Median Blur"]:
                    params = {"ksize": ksize}
                elif operation == "Gaussian Blur":
                    params = {"ksize": ksize, "sigma_x": sigma_x}
                elif operation == "Bilateral Filtering":
                    params = {"d": d, "sigma_color": sigma_color, "sigma_space": sigma_space}
                elif operation in ["Erosion", "Dilation", "Opening", "Closing"]:
                    params = {"iterations": iterations, "kernel_size": kernel_size}
                elif operation == 'Noise Removal':
                    params = {"h": h, "hColor": hColor, "templateWindowSize": templateWindowSize, "searchWindowSize": searchWindowSize}
                elif operation == "Face Detection":
                    params = {"scale": scale, "neighbor": neighbor}
                elif operation == 'Strike Removal':
                    params = {"paint": paint, "neighbor": neighbor, "kernel": kernel, "iteration": iteration}
                elif operation == 'Image Inpainting':
                    params = {"mask": mask_path, "method": inpaint_method, "radius": inpaint_radius}

                if operation in cumulative_operations:
                    generated_code = generate_function_code(operation, params)
                    st.session_state.output += f"\n{generated_code}"
                
            # Always show download button if there's code
            if st.session_state.output:
                st.download_button(
                    label="📄 Download Code",
                    data=st.session_state.output,
                    file_name="pipeline.py",
                    mime="text/plain",
                    use_container_width=True 
                    )

    except Exception as e:
        st.error(f"Error while preparing downloads: {e}")

else:
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h2>🚀 Welcome to Enhanced Image Preprocessing Suite</h2>
        <p style="font-size: 18px; color: #666;">
            Upload an image from the sidebar to get started with various preprocessing operations:
        </p>
    </div>
    """, unsafe_allow_html=True)    
