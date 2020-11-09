from numpy import asarray
import numpy as np
from PIL import Image
img     = Image.open("13_pred.png")
pixels  = asarray(img)
max_val = np.amax(img)
pixels  = Image.fromarray(pixels.astype('float32')/max_val)
print(pixels)
pixels.show()

