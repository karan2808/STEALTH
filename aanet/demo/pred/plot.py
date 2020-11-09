import cv2
import time
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread("136_pred.png")
plt.imshow(img1, cmap = "flag")
plt.show()
plt.savefig('prediction_136.png')
# img1 = cv2.normalize(img1, dst = None, alpha = 0, beta = 65536, norm_type = cv2.NORM_MINMAX)
# cv2.imshow('frame1', img1)
# cv2.waitKey(10000)
# for rows in img1:
#     print(rows)