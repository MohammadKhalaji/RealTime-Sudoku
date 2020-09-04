import cv2
from processInput import *
import time
import matplotlib.pyplot as plt
import neuralnet
import solver

sample = cv2.imread('sample_images/sample.jpeg')
sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
camera = True
cap = cv2.VideoCapture(0)

detected = []
GOOD_TO_GO = 100

while True and camera: 
    ret, frame = cap.read()
    orig = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    preprocessed = preprocess(frame)
    contour, found = find_contours(preprocessed)
    
    if found: 
        if (1 + len(detected)) % 10 == 0:
            print(f'Detected ({1+len(detected)}/{GOOD_TO_GO})')
        cv2.drawContours(orig, [contour], -1, (0, 255, 0), 3)
        warp = make_square(frame, contour)
        detected.append(warp)
        if len(detected) == GOOD_TO_GO: 
            cv2.destroyAllWindows()
            break
    else: 
        # detected = []
        # print("Sudoku frame NOT found...")
        pass
        
    cv2.imshow("frame", orig) 
    del orig  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if not camera: 
    preprocessed = preprocess(sample)
    contour, found = find_contours(preprocessed)
    warp = make_square(sample, contour)
    detected = [warp]
     
   
sharpest = get_sharpest(detected)
cv2.imshow('Sharpest', sharpest)
cv2.waitKey(0)

squares, gridded_image = infer_grid(sharpest)
cv2.imshow('Gridded Image', gridded_image)
cv2.waitKey(0)

clean_squares = finalize(squares, blur=True)
sudoku = np.zeros((9, 9), dtype='uint8')
for square, i, j in clean_squares:
    sudoku[i, j] = neuralnet.predict(square)


print('Here is your board....\n')
solver.print_board(sudoku)
print('______________________________________________________')
print('Here is your solution...\n')
solver.solve(sudoku)
solver.print_board(sudoku)


input()