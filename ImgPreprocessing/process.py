import cv2 as cv
import numpy as np
import cv2 as cv
import numpy as np
from PIL import Image
import io
from matplotlib import pyplot as plt
import os 
import math

##################################################  FUNTIONS ##################################################

# Global Threshold
def global_threshold(img, thresh_type, thresh, maxVal):
    try:
        img_copy = img.copy()
        if len(img.shape) == 3:                                                     # Check if image is grayscale or else convert
            gray_img = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)   
        else:
            gray_img = img
            
        thresh_dict = {
            "Binary": cv.THRESH_BINARY,
            "Binary Inverted": cv.THRESH_BINARY_INV,
            "Truncate": cv.THRESH_TRUNC,
            "To Zero": cv.THRESH_TOZERO,
            "To Zero Inverted": cv.THRESH_TOZERO_INV
        }
        _, thresh_img = cv.threshold(gray_img, thresh, maxVal, thresh_dict[thresh_type])    # Apply global threshold to image
        output = cv.cvtColor(thresh_img , cv.COLOR_GRAY2BGR)                                # Return color image
        return output

    except Exception as e:
        print(f"Error in global_threshold :{e}")
        return None

# Adaptive Mean Threshold
def adaptive_mean_threshold(img, maxVal, thresh_type, blockSize, C):
    try:
        img_copy = img.copy()
        if len(img.shape) == 3:                                             # Check if image is grayscale or else convert
            gray_img = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
        else:
            gray_img = img
        binary_dict = {
            "Binary": cv.THRESH_BINARY,
            "Binary Inverted": cv.THRESH_BINARY_INV
        }
        thresh_img = cv.adaptiveThreshold(gray_img, maxVal, cv.ADAPTIVE_THRESH_MEAN_C, 
                                        binary_dict[thresh_type], blockSize, C)      # Apply adaptive mean threshold to image
        output = cv.cvtColor(thresh_img , cv.COLOR_GRAY2BGR)                         # Return color image
        return output
    
    except Exception as e:
        print(f"Error in adaptive_threshold :{e}")
        return None

# Adaptive Gaussian Threshold
def adaptive_gaussian_threshold(img, maxVal, thresh_type, blockSize, C):
    try:
        img_copy = img.copy()
        if len(img.shape) == 3:                                                       # Check if image is grayscale or else convert
            gray_img = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
        else:
            gray_img = img
        binary_dict = {
            "Binary": cv.THRESH_BINARY,
            "Binary Inverted": cv.THRESH_BINARY_INV
        }
        thresh_img = cv.adaptiveThreshold(gray_img, maxVal, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        binary_dict[thresh_type], blockSize, C)         # Apply adaptive gaussian threshold to image
        output = cv.cvtColor(thresh_img , cv.COLOR_GRAY2BGR)
        return output                                                                   # Return color image

    except Exception as e:
        print(f"Error in adaptive_threshold :{e}")
        return None

# Otsu's Threshold
def otsu_threshold(img, thresh_type, maxVal):
    try:
        img_copy = img.copy()
        if len(img.shape) == 3:                                                     # Check if image is grayscale or else convert
            gray_img = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
        else:
            gray_img = img
        binary_dict = {
            "Binary": cv.THRESH_BINARY,
            "Binary Inverted": cv.THRESH_BINARY_INV
        }
        _, thresh_img = cv.threshold(gray_img, 0, maxVal, 
                                    binary_dict[thresh_type] + cv.THRESH_OTSU)  # Apply adaptive gaussian threshold to image
        output = cv.cvtColor(thresh_img , cv.COLOR_GRAY2BGR)
        return output                                                           # Return color image
    
    except Exception as e:
        print(f"Error in otsu_threshold :{e}")
        return None

#Box Detection
def box_detection(img, width, height, mode, method):
    try:
        img_copy = img.copy()
        mode_dict = {
            "External": cv.RETR_EXTERNAL,
            "List": cv.RETR_LIST,
            "CCOMP": cv.RETR_CCOMP,
            "Tree": cv.RETR_TREE
        }
        method_dict = {
            "None" : cv.CHAIN_APPROX_NONE,
            "Simple" : cv.CHAIN_APPROX_SIMPLE
        }
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)                                              # Convert to grayscale image        

        _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)             # Finding threshold of the image

        contours, hierarchy = cv.findContours(thresh, mode_dict[mode], cv.CHAIN_APPROX_SIMPLE)  # Finding the contours of image

        for cnt in contours:
            approx = cv.approxPolyDP(cnt, 0.02*cv.arcLength(cnt, True), True)                   # Simplify the polygon
            x,y,w,h = cv.boundingRect(cnt)                                                      # Storing the verices of the polygon
            if len(approx) == 4 and (w > width and h > height):                                 # Check if polygon has 4 vertices and its height and width is within the given constraints
                cv.drawContours(img_copy, [cnt], -1, (0,0,255), 2)                              # Drawing these contours onto the image
        return img_copy

    except Exception as e:
        print(f"Error in box_detection :{e}")
        return None

# Skew Correction Using houghLine
def skew_houghLine(image, line_angle, min_line_len, max_line_gap):
    try:
        img = image
        img_copy = img.copy()

        height,width = img.shape[:2]
        center = (width//2, height//2)
        original_image = img.copy()

        if len(img.shape) == 3:                                          # Check if image in grayscale or else convert to grayscale
            gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
        else:
            gray = img

        blur = cv.GaussianBlur(gray, (5, 5), 0)                          # Apply gaussian blur on to the image

        _, threshed = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV+      # Apply threshold to the image
        cv.THRESH_OTSU)

        eroded = cv.erode(threshed,(3,3),1)                              # Apply erosion to the image

        dilate = cv.dilate(eroded, (35, 35), iterations=3)               # Apply dilation to the image

        lines = cv.HoughLinesP(dilate,1,np.pi/180,200,None,
                                minLineLength=min_line_len,maxLineGap=max_line_gap)    # Detect lines using Hough Line Transform
        angles = []
        if lines is not None:
            horizontal_lines = []
            for i,line  in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                diff_x = x2-x1  
                diff_y = y2-y1
                if abs(diff_y) < line_angle and abs(diff_x) != 0:           # Filter out horizontal lines
                    horizontal_lines.append((x1, y1, x2, y2))
                    slope = diff_y / diff_x
                    angle = math.degrees(math.atan(slope))                  # Calculate angles in degrees
                    angles.append(angle)
                    
            for line in horizontal_lines:
                x1, y1, x2, y2 = line
                cv.line(img_copy, (x1,y1), (x2,y2), (0,0,255), 1, cv.LINE_AA)
        if angles:
            rotation_angle = sum(angles) / len(angles)                      # Compute average angle
        else:
            rotation_angle = 0
        rotation_matrix = cv.getRotationMatrix2D(center, rotation_angle, 1)

        rotated_image = cv.warpAffine(original_image, rotation_matrix, (width, height), 
                                    flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)   # Rotate image to correct the skew

        return rotated_image
    
    except Exception as e:
        print(f"Error in skew_houghLine :{e}")
        return None

# Watermark removal
def watermark_remove(img, morph_radius, dilation_iter):
    try:
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        k = morph_radius*2 + 1
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k,k))
        watermark = cv.morphologyEx(src=img, op=cv.MORPH_CLOSE, kernel=kernel)              # Estimate watermark

        (th, mask) = cv.threshold(watermark, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        mask = cv.dilate(mask, kernel=None, iterations=dilation_iter)                       # Calculate mask of the watermark
        mask = mask.astype(bool)

        watermark_value = np.median(img[mask], axis=0)                                      # Estimate the watermark's color

        result = img.copy()
        correction_factor =  np.float32(255 / watermark_value)
        result[mask] = (result[mask] * correction_factor).clip(0, 255).astype('u1')         # Correct watermarked pixels
        return result

    except Exception as e:
        print(f"Error in watermark_remove :{e}")
        return None

# IMAGE INPAINTING
def ImageInpainting(img, mask, method, radius):
    try:
        if mask is not None:                                # Check if mask is valid
            mask_img = np.array(mask)                       # Convert mask to numpy array
            method_flag = cv.INPAINT_TELEA if method == "Telea" else cv.INPAINT_NS   # Select the inpainting method
            inpainted_img = cv.inpaint(img, mask_img, radius, method_flag)              # Apply inpainting to the image
            return inpainted_img
        else:
            return img
    except Exception as e:
        print(f"Error in ImageInpainting: {e}")
        return None  

## LINE FUNCTIONS ##
# Line detection

def line_detection(img, threshold, min_length, max_gap):
    try:
        """Detect and draw all lines"""
        img_copy = img.copy()                            
        gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)    # Convert the image into grayscale
        edges = cv.Canny(gray, 50, 150, apertureSize=3)    # Detect all the edges

        lines = cv.HoughLinesP(                            # Detect all the line using HoughLines
            edges,
            rho=1,
            theta=np.pi/100,
            threshold=threshold,
            minLineLength=min_length,
            maxLineGap=max_gap
        )
        # Draw The Line
        if lines is not None:                   
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2) # Line color:Green
    
        return img_copy
    except Exception as e:
        print(f"Error in line_detection :{e}")
        return None

# Horizontal Line detection
def horizontal_detection(img, threshold, min_length, max_gap):
    try:

        img_copy = img.copy() 
        gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)      # Conver the image into gray scale   
        blur = cv.GaussianBlur(gray, (3, 3), 0)              # Applying Gaussian Blur
        edges = cv.Canny(blur, 50, 150, apertureSize=3)      # Detecting Edges

        lines = cv.HoughLinesP(                              # Detect all the lines using HoughLines
            edges,
            rho=1,
            theta=np.pi/100,
            threshold=threshold,
            minLineLength=min_length,
            maxLineGap=max_gap
        )
        #Draw all the lines
        if lines is not None:       
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y1 - y2) <= 1:        #Check the line is horizontal or not
                    cv.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)  #Line color : green

        return img_copy

    except Exception as e:
        print(f"Error in horizontal_detection :{e}")
        return None

# Vertical Line Detection
def vertical_detection(img, threshold, min_length, max_gap):
    try:
        """Detect vertical lines"""
        img_copy = img.copy()
        gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)               # Convert the image into gray scale
        blur = cv.GaussianBlur(gray, (3, 3), 0)                       # Applying Gaussian blur
        edges = cv.Canny(blur, 50, 150, apertureSize=3)               # Detect all  the edges
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))      # create a vertical kernal to connect vertical edges
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)        # close small gaps in the edge

        lines = cv.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=threshold,
            minLineLength=min_length,
            maxLineGap=max_gap
        )
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))  # Calculate the angle of the line
                if abs(angle - 90) <= 15:
                    cv.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw vertical line

        return img_copy
    except Exception as e:
        print(f"Error in vertical_detection : {e}")
        return None

# Horizontal Line Removal
def horizontal_line_removal(img, c, neighborhood, kernel, iteration):
    try:
        img_copy = img.copy()
        gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)  # Conver the image into grayscale
        thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, neighborhood, c)  # Convert it into Binary image
        combined_mask = np.zeros(gray.shape, np.uint8) # Create an empty mask to store the line 

        horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,kernel)) # Define the kernal for detecting horizontal lines
        horizontal_lines = cv.morphologyEx(thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=iteration) # Extract the horizontal line using the kernal

        cnts_h = cv.findContours(horizontal_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Find boundaries from the horizontal line

        # Draw the countours with combined mask
        cnts_h = cnts_h[0] if len(cnts_h) == 2 else cnts_h[1] 
        for contour in cnts_h:
            cv.drawContours(combined_mask, [contour], -1, (255, 255, 255), 2)

        #Remove the lines from the original image using the combined mask
        img_dst = cv.inpaint(img, combined_mask, 3, cv.INPAINT_TELEA)
        return img_dst
    except Exception as e:
        print("Error in hoorizontal_line_removal : {e}")
        return None

# Vertical Line Removal
def vertical_line_removal(img, c, neighborhood, kernel, iteration):
    try:

        img_copy = img.copy()   #make a copy of the oroginal image
        gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY) # Conver the image into gray scale
        thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, neighborhood, c) # Convert it into Binary image
        combined_mask = np.zeros(gray.shape, np.uint8) # Create an empty mask to store the line 

        vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel,1))  # Define the keranel for detecting the vertical line 
        vertical_lines = cv.morphologyEx(thresh, cv.MORPH_OPEN, vertical_kernel, iterations=iteration) # Extract all the vertical lines

        cnts_v = cv.findContours(vertical_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #Find boundaries from the vertical line
        # Draw the countours for combine mask
        cnts_v = cnts_v[0] if len(cnts_v) == 2 else cnts_v[1] 
        for contour in cnts_v:
            cv.drawContours(combined_mask, [contour], -1, (255, 255, 255), 2)
        # Remove the lines from the original image using combined mask
        img_dst = cv.inpaint(img, combined_mask, 3, cv.INPAINT_TELEA)
        return img_dst
        
    except Exception as e:
        print(f"Error in vertical_line_removal{e}")
        return None

# Dotted Line Detection
def dotted_line_detection(img, min_len, max_len, row_tolerance, col_tolerance):
    try:

        """Detect dotted lines"""
        img_copy = img.copy()                            # Make the copy of the original image
        gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)  # Apply grayscale
        blur = cv.GaussianBlur(gray, (3, 3), 0)          # Apply Gaussian Blur
        edges = cv.Canny(blur, 50, 150, apertureSize=3)  # Find out all the edges of the image

        lines = cv.HoughLinesP(                         #Extract all the lines using HoughLinesp
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=min_len,
            maxLineGap=max_len
        )

        # Draw the Dotted Lines
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

                # Horizontal
                if abs(angle) < 10:
                    if abs(y1 - y2) <= row_tolerance:
                        cv.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Vertical
                elif abs(angle - 90) < 10:
                    if abs(x1 - x2) <= col_tolerance:
                        cv.line(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return img_copy
    except Exception as e:
        print(f"Error in dotted_line_detection :{e}")
        return None

## BLUR FUNCTIONS ##
# Mean Blur 
def mean_blur(img, ksize):
    try:
        """Apply mean blur"""
        return cv.blur(img, (ksize, ksize))
    except Exception as e:
        print(f"Error in mean_blur :{e}")

# Gaussian Blur
def gaussian_blur(img, ksize, sigma_x):
    try:
        """Apply Gaussian blur"""
        return cv.GaussianBlur(img, (ksize, ksize), sigma_x)
    except Exception as e:
        print(f"Error in gaussian_blur :{e}")
        return None

# Median Blur
def median_blur(img, ksize):
    try:
        """Apply median blur"""
        return cv.medianBlur(img, ksize)
    except Exception as e:
        print(f"Error in median_blur")
        return None

# Bilateral Filterign
def bilateral_filtering(img, d, sigma_color, sigma_space):
    try:
        """Apply bilateral filtering"""
        return cv.bilateralFilter(img, d, sigma_color, sigma_space)
    except Exception as e:
        print(f"Error in bilateral_filtering :{e}")

## MORPHOLOGICAL FUNCTIONS ##
# Erosion

def erosion(img, iterations, kernel_size):
    try:
        """Apply erosion"""
        gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)             # Convert the image into grayscale           
        kernel = np.ones((kernel_size, kernel_size), np.uint8)       # Define the kernel size for apply eroison
        eroded = cv.erode(gray_image, kernel, iterations=iterations) # Erode the image                            
        return cv.cvtColor(eroded, cv.COLOR_GRAY2BGR)
    except Exception as e:
        print(f"Error in erosion: {e}")
        return None

# Dilation
def dilation(img, iterations, kernel_size):
    """Apply dilation"""
    try:
        gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)               # Convert the image into grayscale                    
        kernel = np.ones((kernel_size, kernel_size), np.uint8)         # Define the kernel size for apply dilation             
        dilated = cv.dilate(gray_image, kernel, iterations=iterations) # Dilate the image                                        
        return cv.cvtColor(dilated, cv.COLOR_GRAY2BGR)                   
    except Exception as e:
        print(f"Error in dilation: {e}")
        return None

# Opening
def opening(img, iterations, kernel_size):
    """Apply opening"""
    try:
        gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)                                   # Convert the image into grayscale                                                               
        kernel = np.ones((kernel_size, kernel_size), np.uint8)                             # Define the kernel size for apply eroison                                                                      
        opened = cv.morphologyEx(gray_image, cv.MORPH_OPEN, kernel, iterations=iterations) # Applay Morphology Open                                                                                                       
        return cv.cvtColor(opened, cv.COLOR_GRAY2BGR)
    except Exception as e:
        print(f"Error in opening: {e}")
        return None

# Closing
def closing(img, iterations, kernel_size):
    """Apply closing"""
    try:
        gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)                                    # Convert the image into grayscale                                                                                                        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)                              # Define the kernel size for apply eroison                                                                             
        closed = cv.morphologyEx(gray_image, cv.MORPH_CLOSE, kernel, iterations=iterations) # Apply Morphology Close                                                                                                           
        return cv.cvtColor(closed, cv.COLOR_GRAY2BGR)
    except Exception as e:
        print(f"Error in closing: {e}")
        return None

# Noise Removal
def NoiseRemoval(img, h, hColor, windowsize, searchwindowsize):
    try:
        img_copy = img.copy()
        gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)               # Convert into gray scale
        gaussian_blur = cv.GaussianBlur(gray, (5,5), 6)               # Apply gaussian blur
        media_blur = cv.medianBlur(gray, 5)                           # Apply Median blur 
        bilateral_filter = cv.bilateralFilter(img_copy, 9, 75, 75)    # Apply Median BLur

        # denoising of image saving it into denoise
        denoise = cv.fastNlMeansDenoisingColored(img_copy, None, h=h, hColor=hColor, templateWindowSize=windowsize, searchWindowSize=searchwindowsize)

        return denoise 
    except Exception as e:
        print(f"Error in NoiseRemoval: {e}")
        return None

#face detection
def face_detect(img , scale , neighbor):
    img_copy = img.copy()               #copy the image
    gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY) #convert it into grayscale
    face_cas = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml') #load haarcascade file
    faces = face_cas.detectMultiScale(gray , scaleFactor=scale , minNeighbors=neighbor) #Detect faces from the image
    
    # Draw rectangle on detected face
    for (x,y,w,h) in faces:
        cv.rectangle(img_copy , (x,y) , (x+w , y+h) , (0,255,0) , 2)

    #cv.putText(img_copy, f"{scale}", (50, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img_copy

# Strike Removal
def LineOverText(img, paint, neighborhood, kernel, iteration ):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #convert the imaeg into grayscale
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, neighborhood, 21) #convert it into binary image
    combined_mask = np.zeros(gray.shape, np.uint8) # Create a mask 
    
    #find out the horizontal keraels 
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel, 1))
    horizontal_lines = cv.morphologyEx(thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=iteration) #Find the horizontal lines
    
    # extract the contours from the horizzontal line
    cnts_h ,_= cv.findContours(horizontal_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    #Extract the text from the image
    text_kernel = cv.getStructuringElement(cv.MORPH_RECT , (5,5))
    text_dilated = cv.dilate(thresh , text_kernel , iterations = 3) # Apply dialtion

    # Mask the Line over text 
    for contour in cnts_h :
        x,y,w,h = cv.boundingRect(contour)
        line_mask = np.zeros(gray.shape , dtype=np.uint8)
        cv.drawContours(line_mask , [contour] ,-1,255-1)

        intersection = cv.bitwise_and(text_dilated , line_mask)
        if cv.countNonZero(intersection) > 0:   
            cv.drawContours(combined_mask , [contour] , -1 , 255 , 1)

        #Remove the line using inpaint
        img_dst = cv.inpaint(img ,combined_mask , paint , cv.INPAINT_TELEA)
    return img_dst