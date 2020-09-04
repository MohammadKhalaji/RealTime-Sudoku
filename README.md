# RealTime-Sudoku
Real-time Sudoku solver with OpenCV and PyTorch
1. run `main.py`

2. Using OpenCV Python, the sudoku square is identified: 

<!-- ![Sudoku Square Identification]() -->
<img align="center" width="100" height="100" src="http://www.fillmurray.com/100/100">

![Sudoku Square Identification](https://github.com/MohammadKhalaji/RealTime-Sudoku/blob/master/readme_images/2.jpg)

3. Using morphological operations and thresholding, the values inside the tiles are cleaned from noise and irrelevant lines.

4. The output of the previous step is fed to a convolutional neural network, trained on MNIST, using PyTorch. 

![ConvNet labels the tiles](https://github.com/MohammadKhalaji/RealTime-Sudoku/blob/master/readme_images/3.jpg)

5. The puzzle is solved :)

![Final](https://github.com/MohammadKhalaji/RealTime-Sudoku/blob/master/readme_images/4.jpg)
