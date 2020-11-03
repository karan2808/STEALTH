import cv2
import numpy as np 
import matplotlib.pyplot as plt

img1 = cv2.imread('OUTDOOR/CARS/scene5/rectified/thermal/left_thermal_default.png', cv2.IMREAD_UNCHANGED)
disp = 'OUTDOOR/CARS/scene10/gt_disparity/thermal/gt_disparity_interp.txt'
disp_range = 'OUTDOOR/CARS/scene5/gt_disparity/thermal/disp_range.txt'
#print(img1.shape)

disp_vals = open(disp_range,'r')
disp_vals = disp_vals.readlines()
disp_vals = disp_vals[0].split('\n')
disp_vals = disp_vals[0].split(',')
#disp_vals = disp_vals
print(disp_vals)

my_disp_file = open(disp, "r")
my_disp_file = my_disp_file.readlines()
my_disp_mat  = []

for line in my_disp_file:
    line_arr = line.split('\n')
    line_arr = line_arr[0].split(',')
    my_disp_mat.append(line_arr)

my_disp_mat = np.array(my_disp_mat)
print(my_disp_mat.shape)
my_disp_mat = my_disp_mat.astype('float')
#my_disp_mat = np.clip(my_disp_mat,int(disp_vals[0]),int(disp_vals[1]))
#cv2.imshow('frame', (my_disp_mat + 60)/np.amax(my_disp_mat + 60))
plt.imsave('test.png', my_disp_mat,cmap = 'gray')
plt.imshow(my_disp_mat, cmap = "gray")
plt.show()

img = cv2.imread('test.png',0)
cv2.imshow('frame', img)
cv2.imwrite('10.png',img)
hist_full = cv2.calcHist(img)
plt.plot(hist_full)
plt.show()
#cv2.waitKey(10000)