import cv2 
import numpy as np
import imutils
import matplotlib.pyplot as plt 

INTERMEDIARY_SHAPE = (200, 200)
MNIST_SHAPE = (28, 28)


def show(image): 
    cv2.imshow("Game Boy Screen", image)
    cv2.waitKey(0)

def dist(x, y): 
    return np.sqrt(np.sum((x - y) ** 2))

def preprocess(image):
    # image = img.copy()
    # blur for less noise
    image = cv2.GaussianBlur(image, (9, 9), 0)
    # can't use absolute threshold values here! need adaptive threshold
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # invert
    image = cv2.bitwise_not(image, image)
    # dilate to remove small noises
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    image = cv2.dilate(image, kernel)
    # Canny edge detection for better contour finding
    image = cv2.Canny(image, 30, 200)
    
    return image

def find_contours(image): 
    # image = img.copy()
    
    contours = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    main_contour = None
    for contour in contours: 
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        # if has 4 points! 
        if len(approx) == 4:
            main_contour = approx
            break
    if main_contour is None: 
        return None, False
    cntArea = cv2.contourArea(main_contour)
    imgArea = image.shape[0] * image.shape[1]
    if cntArea > 0.07 * imgArea: # if it is big enough!
        return main_contour, True
    return None, False

def make_square(img, contour): 
    image = img.copy()
    points = contour.reshape((4, 2))
    points_sorted = np.zeros((4, 2))    
    sum = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    tl = points_sorted[0] = points[np.argmin(sum)] # top left
    tr = points_sorted[1] = points[np.argmin(diff)] # top right
    br = points_sorted[2] = points[np.argmax(sum)] # bottom right
    bl = points_sorted[3] = points[np.argmax(diff)] # bottom left

    max_width = int(max(dist(br, bl), dist(tr, tl))) + 1
    max_height = int(max(dist(tr, br), dist(tl, bl))) + 1
    max_size = max(max_height, max_width)
    dest = np.array([
        [0, 0], 
        [max_size - 1, 0], 
        [max_size - 1, max_size - 1], 
        [0, max_size - 1]
    ])

    M = cv2.getPerspectiveTransform(points_sorted.astype('float32'), dest.astype('float32'))

    warp = cv2.warpPerspective(image, M, (max_size, max_size))

    h, w = warp.shape
    k = 0
    if h % 9 != 0: 
        k = 9 - (h % 9)
    return cv2.resize(warp, (h+k, w+k))

def get_sharpest(detected): 
    max = -np.inf
    chosen = None
    for image in detected: 
        variance = cv2.Laplacian(image, cv2.CV_64F).var()
        if variance > max: 
            max = variance
            chosen = image
    if chosen.shape[0] > 300: 
        chosen = cv2.GaussianBlur(chosen, (9, 9), 0)
    

    return chosen

def infer_grid(image): 
    # h, w = image.shape
    # if w % 9 != 0: 
    #     k = 9 - (w % 9)
    # image = cv2.resize(image, (w+k, w+k))
    gridded_image = image.copy()
    gridded_image = cv2.cvtColor(gridded_image, cv2.COLOR_GRAY2BGR)
    max_size = image.shape[0]
    sq_size = int(max_size / 9)
    squares = []
    for i in range(9): 
        y1 = i * sq_size
        y2 = (i + 1) * sq_size
        for j in range(9):
            x1 = j * sq_size
            x2 = (j + 1) * sq_size
            top_left = (x1, y1)
            bottom_right = (x2, y2)
            roi = image[y1:y1+sq_size, x1:x1+sq_size]
            cv2.rectangle(gridded_image, top_left, bottom_right, (0, 255, 0), 3)
            squares.append(roi)
    return squares, gridded_image
    
def pad(img, target_shape): 
    res = np.ones(target_shape, dtype='uint8') * 255
    h, w = img.shape
    cy, cx = target_shape[0] // 2, target_shape[1] // 2
    res[cy-(h//2):cy-(h//2)+h, cx-(w//2):cx-(w//2)+w] = img
    return res

def trim_borders(square, iterations=1):
    # remove black borders from the sides
    original_shape = square.shape
    image = square.copy()
    for _ in range(iterations): 
        h, w = image.shape
        max_destroy = 1 + (h // 8)
        rs, re, cs, ce = 0, h, 0, w
        for i in range(max_destroy): 
            rs_blacks = np.sum(255 - image[i]) // 255
            if rs_blacks > 0.5 * w: 
                rs = i + 1
            cs_blacks = np.sum(255 - image[:, i]) // 255
            if cs_blacks > 0.5 * w: 
                cs = i + 1
            j = -(i + 1)
            re_blacks = np.sum(255 - image[j]) // 255
            if re_blacks > 0.5 * w: 
                re = j
            ce_blacks = np.sum(255 - image[:, j]) // 255
            if ce_blacks > 0.5 * w: 
                ce = j
                
        image = image[rs:re, cs:ce]
    return pad(image, original_shape)

def is_empty(square):
    contours, hierarchy = cv2.findContours(square, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    if len(contours) == 1: # if only a huge contour is available, the square is emplty
        return True
    unique, counts = np.unique(square, return_counts=True)
    czip = dict(zip(unique, counts))
    if 0 not in czip: 
        czip[0] = 0
    ratio = czip[0] / (czip[0] + czip[255])
    if ratio < 0.05: # if the ratio of black pixels is too small, the square is empty
        return True
    return False

def mask_the_digit(square): 
    contours, hierarchy = cv2.findContours(square, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    digit = contours[1] # the second largest contour is the number
    x, y, w, h = cv2.boundingRect(digit)
    rect = square[y:y+h, x:x+w]
    return pad(rect, square.shape)

def finalize(squares, blur=False):
    clean_squares = []
    for idx in range(len(squares)):
        i = idx // 9
        j = idx % 9
        square = squares[idx]  
        square = cv2.adaptiveThreshold(square, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)      
        prep = trim_borders(square, iterations=3)
        if is_empty(prep):
            prep = np.ones(prep.shape).astype('uint8') * 255
        else:
            prep = mask_the_digit(prep)
            prep = cv2.resize(prep, INTERMEDIARY_SHAPE, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            se5 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
            se3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            # prep = cv2.morphologyEx(prep, cv2.MORPH_DILATE, se5)
            prep = cv2.morphologyEx(prep, cv2.MORPH_DILATE, se5)
            prep = cv2.morphologyEx(prep, cv2.MORPH_DILATE, se3)
            prep = cv2.morphologyEx(prep, cv2.MORPH_ERODE, se5)
            prep = cv2.morphologyEx(prep, cv2.MORPH_ERODE, se3)
            if blur:
                prep = cv2.GaussianBlur(prep, (5, 5), 0)
            prep = cv2.resize(prep, MNIST_SHAPE, interpolation=cv2.INTER_NEAREST)

            clean_squares.append((prep, i, j))
    return clean_squares