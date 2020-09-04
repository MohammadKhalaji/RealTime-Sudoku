from CNN.net import Net
import torch
import os
import cv2

model = Net()
model.load_state_dict(torch.load('CNN/cnn.pt'))
model.eval()
model = model.float()
MU = 0.1307
SIGMA = 0.3081

def predict(square): 
    square = square / 255
    square = 1 - square
    square = (square - MU) / SIGMA
    square = torch.from_numpy(square)
    with torch.no_grad(): 
        square = square.view(1, 1, 28, 28)
        output = model(square.float())
        pred = 1 + output.argmax(dim=1, keepdim=True)
    return pred