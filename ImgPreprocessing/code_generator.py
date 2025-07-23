##################################################  CODE GENERATION ##################################################

def generate_function_code(operation, params):
    try:
        code_templates = {
        "Global Threshold": f"""
# Global Threshold
def global_threshold(img, thresh_type="{params.get('thresh_type', 'Binary')}", thresh={params.get('thresh', 125)}, maxVal={params.get('maxVal', 255)}):
    img_copy = img.copy()
    if len(img.shape) == 3:
        gray_img = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
    else:
        gray_img = img
        
    thresh_dict = {{
        "Binary": cv.THRESH_BINARY,
        "Binary Inverted": cv.THRESH_BINARY_INV,
        "Truncate": cv.THRESH_TRUNC,
        "To Zero": cv.THRESH_TOZERO,
        "To Zero Inverted": cv.THRESH_TOZERO_INV
    }}
    _, thresh_img = cv.threshold(gray_img, thresh, maxVal, thresh_dict[thresh_type])
    output = cv.cvtColor(thresh_img, cv.COLOR_GRAY2BGR)
    return output

img = global_threshold(img, "{params.get('thresh_type', 'Binary')}", {params.get('thresh', 125)}, {params.get('maxVal', 255)})
""",

        "Adaptive Threshold (Mean)": f"""
# Adaptive Mean Threshold
def adaptive_mean_threshold(img, maxVal={params.get('maxVal', 255)}, thresh_type="{params.get('thresh_type', 'Binary')}", blockSize={params.get('blockSize', 11)}, C={params.get('C', 2)}):
    img_copy = img.copy()
    if len(img.shape) == 3:
        gray_img = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
    else:
        gray_img = img
    binary_dict = {{
        "Binary": cv.THRESH_BINARY,
        "Binary Inverted": cv.THRESH_BINARY_INV
    }}
    thresh_img = cv.adaptiveThreshold(gray_img, maxVal, cv.ADAPTIVE_THRESH_MEAN_C, binary_dict[thresh_type], blockSize, C)
    output = cv.cvtColor(thresh_img, cv.COLOR_GRAY2BGR)
    return output

img = adaptive_mean_threshold(img, {params.get('maxVal', 255)}, "{params.get('thresh_type', 'Binary')}", {params.get('blockSize', 11)}, {params.get('C', 2)})
""",

        "Adaptive Threshold (Gaussian)": f"""
# Adaptive Gaussian Threshold
def adaptive_gaussian_threshold(img, maxVal={params.get('maxVal', 255)}, thresh_type="{params.get('thresh_type', 'Binary')}", blockSize={params.get('blockSize', 11)}, C={params.get('C', 2)}):
    img_copy = img.copy()
    if len(img.shape) == 3:
        gray_img = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
    else:
        gray_img = img
    binary_dict = {{
        "Binary": cv.THRESH_BINARY,
        "Binary Inverted": cv.THRESH_BINARY_INV
    }}
    thresh_img = cv.adaptiveThreshold(gray_img, maxVal, cv.ADAPTIVE_THRESH_GAUSSIAN_C, binary_dict[thresh_type], blockSize, C)
    output = cv.cvtColor(thresh_img, cv.COLOR_GRAY2BGR)
    return output

img = adaptive_gaussian_threshold(img, {params.get('maxVal', 255)}, "{params.get('thresh_type', 'Binary')}", {params.get('blockSize', 11)}, {params.get('C', 2)})
""",

        "Otsu's Threshold": f"""
# Otsu's Threshold
def otsu_threshold(img, thresh_type="{params.get('thresh_type', 'Binary')}", maxVal={params.get('maxVal', 255)}):
    img_copy = img.copy()
    if len(img.shape) == 3:
        gray_img = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
    else:
        gray_img = img
    binary_dict = {{
        "Binary": cv.THRESH_BINARY,
        "Binary Inverted": cv.THRESH_BINARY_INV
    }}
    _, thresh_img = cv.threshold(gray_img, 0, maxVal, binary_dict[thresh_type] + cv.THRESH_OTSU)
    output = cv.cvtColor(thresh_img, cv.COLOR_GRAY2BGR)
    return output

img = otsu_threshold(img, "{params.get('thresh_type', 'Binary')}", {params.get('maxVal', 255)})
""",

        "Skew Correction": f"""
# Skew Correction Using HoughLine
def skew_houghLine(image, line_angle={params.get('line_angle', 10)}, min_line_len={params.get('min_line_len', 100)}, max_line_gap={params.get('max_line_gap', 10)}):
    img = image
    img_copy = img.copy()

    height,width = img.shape[:2]
    center = (width//2, height//2)
    original_image = img.copy()

    if len(img.shape) == 3:
        gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
    else:
        gray = img

    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, threshed = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    eroded = cv.erode(threshed,(3,3),1)
    dilate = cv.dilate(eroded, (35, 35), iterations=3)

    lines = cv.HoughLinesP(dilate,1,np.pi/180,200,None,minLineLength=min_line_len,maxLineGap=max_line_gap)
    angles = []
    if lines is not None:
        horizontal_lines = []
        for i,line  in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            cv.line(img_copy, (x1,y1), (x2,y2), (0, 0, 255), 1, cv.LINE_AA)
            diff_x = x2-x1
            diff_y = y2-y1
            if abs(diff_y) < line_angle and abs(diff_x) != 0:
                horizontal_lines.append((x1, y1, x2, y2))
                slope = diff_y / diff_x
                angle = math.degrees(math.atan(slope))
                angles.append(angle)
                
        for line in horizontal_lines:
            x1, y1, x2, y2 = line
            cv.line(img_copy, (x1,y1), (x2,y2), (0,0,255), 1, cv.LINE_AA)
    if angles:
        rotation_angle = sum(angles) / len(angles)
    else:
        rotation_angle = 0
    rotation_matrix = cv.getRotationMatrix2D(center, rotation_angle, 1)
    rotated_image = cv.warpAffine(original_image, rotation_matrix, (width, height), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return rotated_image

img = skew_houghLine(img, {params.get('line_angle', 10)}, {params.get('min_line_len', 100)}, {params.get('max_line_gap', 10)})
""",

        "Box Detection": f"""
# Box Detection
def box_detection(img, width={params.get('width', 50)}, height={params.get('height', 50)}, mode="{params.get('mode', 'External')}", method="{params.get('method', 'Simple')}"):
    img_copy = img.copy()
    mode_dict = {{
        "External": cv.RETR_EXTERNAL,
        "List": cv.RETR_LIST,
        "CCOMP": cv.RETR_CCOMP,
        "Tree": cv.RETR_TREE
    }}
    method_dict = {{
        "None" : cv.CHAIN_APPROX_NONE,
        "Simple" : cv.CHAIN_APPROX_SIMPLE
    }}
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(thresh, mode_dict[mode], cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv.approxPolyDP(cnt, 0.02*cv.arcLength(cnt, True), True)
        x,y,w,h = cv.boundingRect(cnt)
        if len(approx) == 4 and (w > width and h > height):
            cv.drawContours(img_copy, [cnt], -1, (0,0,255), 2)
    return img_copy

img = box_detection(img, {params.get('width', 50)}, {params.get('height', 50)}, "{params.get('mode', 'External')}", "{params.get('method', 'Simple')}")
""",

        "Watermark Removal": f"""
# Watermark Removal
def watermark_remove(img, morph_radius={params.get('morph_radius', 5)}, dilation_iter={params.get('dilation_iter', 3)}):
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    k = morph_radius*2 + 1
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k,k))
    watermark = cv.morphologyEx(src=img, op=cv.MORPH_CLOSE, kernel=kernel)

    (th, mask) = cv.threshold(watermark, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    mask = cv.dilate(mask, kernel=None, iterations=dilation_iter)
    mask = mask.astype(bool)

    watermark_value = np.median(img[mask], axis=0)
    result = img.copy()
    correction_factor =  np.float32(255 / watermark_value)
    result[mask] = (result[mask] * correction_factor).clip(0, 255).astype('u1')
    return result

img = watermark_remove(img, {params.get('morph_radius', 5)}, {params.get('dilation_iter', 3)})
""",

        "Line Detection": f"""
# Line Detection
def line_detection(img, threshold={params.get('threshold', 100)}, min_length={params.get('min_length', 50)}, max_gap={params.get('max_gap', 10)}):
    img_copy = img.copy()                            
    gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    
    lines = cv.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/100,
        threshold=threshold,
        minLineLength=min_length,
        maxLineGap=max_gap
    )

    if lines is not None:                   
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img_copy

img = line_detection(img, {params.get('threshold', 100)}, {params.get('min_length', 50)}, {params.get('max_gap', 10)})
""",

        "Horizontal Line Detection": f"""
# Horizontal Line Detection
def horizontal_detection(img, threshold={params.get('threshold', 100)}, min_length={params.get('min_length', 50)}, max_gap={params.get('max_gap', 10)}):
    img_copy = img.copy() 
    gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    edges = cv.Canny(blur, 50, 150, apertureSize=3)
    
    lines = cv.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/100,
        threshold=threshold,
        minLineLength=min_length,
        maxLineGap=max_gap
    )

    if lines is not None:       
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) <= 1:
                cv.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img_copy

img = horizontal_detection(img, {params.get('threshold', 100)}, {params.get('min_length', 50)}, {params.get('max_gap', 10)})
""",

        "Vertical Line Detection": f"""
# Vertical Line Detection
def vertical_detection(img, threshold={params.get('threshold', 100)}, min_length={params.get('min_length', 50)}, max_gap={params.get('max_gap', 10)}):
    img_copy = img.copy()
    gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    edges = cv.Canny(blur, 50, 150, apertureSize=3)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    
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
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            if abs(angle - 90) <= 15:
                cv.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img_copy

img = vertical_detection(img, {params.get('threshold', 100)}, {params.get('min_length', 50)}, {params.get('max_gap', 10)})
""",

        "Horizontal Line Removal": f"""
# Horizontal Line Removal
def horizontal_line_removal(img, c={params.get('c', 2)}, neighborhood={params.get('neighborhood', 11)}, kernel={params.get('kernel', 30)}, iteration={params.get('iteration', 3)}):
    img_copy = img.copy()
    gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, neighborhood, c)
    combined_mask = np.zeros(gray.shape, np.uint8)
    
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,kernel))
    horizontal_lines = cv.morphologyEx(thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=iteration)
    
    cnts_h = cv.findContours(horizontal_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts_h = cnts_h[0] if len(cnts_h) == 2 else cnts_h[1] 
    for contour in cnts_h:
        cv.drawContours(combined_mask, [contour], -1, (255, 255, 255), 2)

    img_dst = cv.inpaint(img, combined_mask, 3, cv.INPAINT_TELEA)
    return img_dst

img = horizontal_line_removal(img, {params.get('c', 2)}, {params.get('neighborhood', 11)}, {params.get('kernel', 30)}, {params.get('iteration', 3)})
""",

        "Vertical Line Removal": f"""
# Vertical Line Removal
def vertical_line_removal(img, c={params.get('c', 2)}, neighborhood={params.get('neighborhood', 11)}, kernel={params.get('kernel', 30)}, iteration={params.get('iteration', 3)}):
    img_copy = img.copy()
    gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, neighborhood, c)
    combined_mask = np.zeros(gray.shape, np.uint8)
    
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel,1))
    vertical_lines = cv.morphologyEx(thresh, cv.MORPH_OPEN, vertical_kernel, iterations=iteration)
    
    cnts_v = cv.findContours(vertical_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts_v = cnts_v[0] if len(cnts_v) == 2 else cnts_v[1] 
    for contour in cnts_v:
        cv.drawContours(combined_mask, [contour], -1, (255, 255, 255), 2)

    img_dst = cv.inpaint(img, combined_mask, 3, cv.INPAINT_TELEA)
    return img_dst

img = vertical_line_removal(img, {params.get('c', 2)}, {params.get('neighborhood', 11)}, {params.get('kernel', 30)}, {params.get('iteration', 3)})
""",

        "Dotted Line Detection": f"""
# Dotted Line Detection
def dotted_line_detection(img, min_len={params.get('min_len', 5)}, max_len={params.get('max_len', 15)}, row_tolerance={params.get('row_tolerance', 5)}, col_tolerance={params.get('col_tolerance', 5)}):
    img_copy = img.copy()
    gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    edges = cv.Canny(blur, 50, 150, apertureSize=3)
    
    lines = cv.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=min_len,
        maxLineGap=max_len
    )
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            if abs(angle) < 10:
                if abs(y1 - y2) <= row_tolerance:
                    cv.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif abs(angle - 90) < 10:
                if abs(x1 - x2) <= col_tolerance:
                    cv.line(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img_copy

img = dotted_line_detection(img, {params.get('min_len', 5)}, {params.get('max_len', 15)}, {params.get('row_tolerance', 5)}, {params.get('col_tolerance', 5)})
""",

        "Mean Blur": f"""
# Mean Blur
def mean_blur(img, ksize={params.get('ksize', 5)}):
    return cv.blur(img, (ksize, ksize))

img = mean_blur(img, {params.get('ksize', 5)})
""",

        "Gaussian Blur": f"""
# Gaussian Blur
def gaussian_blur(img, ksize={params.get('ksize', 5)}, sigma_x={params.get('sigma_x', 0)}):
    return cv.GaussianBlur(img, (ksize, ksize), sigma_x)

img = gaussian_blur(img, {params.get('ksize', 5)}, {params.get('sigma_x', 0)})
""",

        "Median Blur": f"""
# Median Blur
def median_blur(img, ksize={params.get('ksize', 5)}):
    return cv.medianBlur(img, ksize)

img = median_blur(img, {params.get('ksize', 5)})
""",

        "Bilateral Filtering": f"""
# Bilateral Filtering
def bilateral_filtering(img, d={params.get('d', 9)}, sigma_color={params.get('sigma_color', 75)}, sigma_space={params.get('sigma_space', 75)}):
    return cv.bilateralFilter(img, d, sigma_color, sigma_space)

img = bilateral_filtering(img, {params.get('d', 9)}, {params.get('sigma_color', 75)}, {params.get('sigma_space', 75)})
""",

        "Erosion": f"""
# Erosion
def erosion(img, iterations={params.get('iterations', 1)}, kernel_size={params.get('kernel_size', 5)}):
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv.erode(gray_image, kernel, iterations=iterations)
    return cv.cvtColor(eroded, cv.COLOR_GRAY2BGR)

img = erosion(img, {params.get('iterations', 1)}, {params.get('kernel_size', 5)})
""",

        "Dilation": f"""
# Dilation
def dilation(img, iterations={params.get('iterations', 1)}, kernel_size={params.get('kernel_size', 5)}):
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv.dilate(gray_image, kernel, iterations=iterations)
    return cv.cvtColor(dilated, cv.COLOR_GRAY2BGR)

img = dilation(img, {params.get('iterations', 1)}, {params.get('kernel_size', 5)})
""",

        "Opening": f"""
# Opening
def opening(img, iterations={params.get('iterations', 1)}, kernel_size={params.get('kernel_size', 5)}):
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv.morphologyEx(gray_image, cv.MORPH_OPEN, kernel, iterations=iterations)
    return cv.cvtColor(opened, cv.COLOR_GRAY2BGR)

img = opening(img, {params.get('iterations', 1)}, {params.get('kernel_size', 5)})
""",

        "Closing": f"""
# Closing
def closing(img, iterations={params.get('iterations', 1)}, kernel_size={params.get('kernel_size', 5)}):
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv.morphologyEx(gray_image, cv.MORPH_CLOSE, kernel, iterations=iterations)
    return cv.cvtColor(closed, cv.COLOR_GRAY2BGR)

img = closing(img, {params.get('iterations', 1)}, {params.get('kernel_size', 5)})
""",

        "Noise Removal": f"""
# Noise Removal
def NoiseRemoval(img, h={params.get('h', 10)}, hColor={params.get('hColor', 10)}, windowsize={params.get('windowsize', 7)}, searchwindowsize={params.get('searchwindowsize', 21)}):
    img_copy = img.copy()
    gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
    gaussian_blur = cv.GaussianBlur(gray, (5,5), 6)
    media_blur = cv.medianBlur(gray, 5)
    bilateral_filter = cv.bilateralFilter(img_copy, 9, 75, 75)
    denoise = cv.fastNlMeansDenoisingColored(img_copy, None, h=h, hColor=hColor, templateWindowSize=windowsize, searchWindowSize=searchwindowsize)
    return denoise

img = NoiseRemoval(img, {params.get('h', 10)}, {params.get('hColor', 10)}, {params.get('windowsize', 7)}, {params.get('searchwindowsize', 21)})
""" ,
    
      "Face Detection" :f"""

def face_detect(img , scale={params.get('scale',1.1)} , neighbor={params.get('neighbor',5)}):
    img_copy = img.copy()               #copy the image
    gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY) #convert it into grayscale
    face_cas = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml') #load haarcascade file
    faces = face_cas.detectMultiScale(gray , scaleFactor=scale , minNeighbors=n) #Detect faces from the image
    # Draw rectangle on detected face
    for (x,y,w,h) in faces:
        cv.rectangle(img_copy , (x,y) , (x+w , y+h) , (0,255,0) , 2)
    
    return img_copy

img = face_detect(img , {params.get('scale',1.1) , {params.get("neighbor" , 5)}}) 
    """ ,
    "Strike Removal" :f"""
    #Line over text
def LineOverText(img, c={params.get('paint', 3)}, neighborhood={params.get('neighborhood' ,  5)}, kernel = {params.get('kernel' , 5)}, iteration={params.get('iteration' , 5)}):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, neighborhood, 21)
    combined_mask = np.zeros(gray.shape, np.uint8)
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel, 1))
    horizontal_lines = cv.morphologyEx(thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=iteration)
    cnts_h ,_= cv.findContours(horizontal_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    text_kernel = cv.getStructuringElement(cv.MORPH_RECT , (5,5))
    text_dilated = cv.dilate(thresh , text_kernel , iterations = 3)
    for contour in cnts_h :
        x,y,w,h = cv.boundingRect(contour)
        line_mask = np.zeros(gray.shape , dtype=np.uint8)
        cv.drawContours(line_mask , [contour] ,-1,255-1)
        intersection = cv.bitwise_and(text_dilated , line_mask)
        if cv.countNonZero(intersection) > 0:   
            cv.drawContours(combined_mask , [contour] , -1 , 255 , 1)
        img_dst = cv.inpaint(img ,combined_mask , 3 , cv.INPAINT_TELEA)
    return img_dst
        
img =LineOverText(img, c={params.get('paint', 3)}, neighborhood={params.get('neighborhood' ,  5)}, kernel = {params.get('kernel' , 5)}, iteration={params.get('iteration' , 5)}) 
    """,
    "Image Inpainting" :f"""

def ImageInpainting(img , mask='{params.get('mask', None)}' , method={params.get("method" , 'TELEA')}, radius={params.get("radius", 3)}):
    if mask is not None:                                # Check if mask is valid
            mask_img = np.array(mask)                       # Convert mask to numpy array
            method_flag = cv.INPAINT_TELEA if method == "Telea" else cv.INPAINT_NS   # Select the inpainting method
            inpainted_img = cv.inpaint(img, mask_img, radius, method_flag)              # Apply inpainting to the image
            return inpainted_img
        else:
            return img

img = ImageInpainting(img , mask='{params.get('mask', None)}' , method={params.get("method" , 'TELEA')}, radius={params.get("radius", 3)}) 
    """ ,
    
    }
        
        return code_templates.get(operation, f"# {operation}")
    
    except Exception as e:
        print(f"Error in code generation : {e}")