import torch
import torchvision
import cv2
import matplotlib
import numpy
import pandas
import sklearn
import tqdm

print('torch      :', torch.__version__)
print('torchvision:', torchvision.__version__)
print('opencv     :', cv2.__version__)
print('matplotlib :', matplotlib.__version__)
print('numpy      :', numpy.__version__)
print('pandas     :', pandas.__version__)
print('sklearn    :', sklearn.__version__)
print('tqdm       :', tqdm.__version__)
print('GPU        :', torch.cuda.is_available())
print('All done')