import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('1_pred.png', cv2.IMREAD_UNCHANGED)
img1 = plt.imshow(img1, cmap='jet')
plt.show()
