```py
import numpy as np
import cv2
from dataset_support import sin_patern
img = np.array(cv2.imread("1.png",cv2.IMREAD_GRAYSCALE)).astype(np.float32)/255
img = sin_patern(img,shape_sin=100,alpha=0.03,vertical=True,bias=1)
cv2.imwrite("2.png",img*255)

```
