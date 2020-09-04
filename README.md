# RealTime-Sudoku
Real-time Sudoku solver with OpenCV and PyTorch
1. run `main.py`

2. Using OpenCV Python, the sudoku square is identified: 

<img align="center" width="300" src="https://github.com/MohammadKhalaji/RealTime-Sudoku/blob/master/readme_images/1.jpg">

<img align="center" width="300" src="https://github.com/MohammadKhalaji/RealTime-Sudoku/blob/master/readme_images/2.jpg">
3. Using morphological operations and thresholding, the values inside the tiles are cleaned from noise and irrelevant lines.

4. The output of the previous step is fed to a convolutional neural network, trained on MNIST, using PyTorch. 

<img align="center" width="300" src="https://github.com/MohammadKhalaji/RealTime-Sudoku/blob/master/readme_images/3.jpg">
5. The puzzle is solved :)

<img align="center" width="300" src="https://github.com/MohammadKhalaji/RealTime-Sudoku/blob/master/readme_images/4.jpg">