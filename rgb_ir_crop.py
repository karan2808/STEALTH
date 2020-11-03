import numpy as np
import cv2 as cv

# Open text file containing paths to all images:
f = open('CMU_RGB.txt','r')
lines = f.readlines()


# Read image pairs and store them in a tuple:
img_pairs = []
for line in lines:
    line = line.split('\n')
    rgb = line[0][68:80]
    ir = line[0][16:28]
    img_pairs.append((rgb,ir))

# Process every image pair:

for pair in img_pairs:

    # Consider only image names starting with '18':
    if pair[0][0:2] == '18':

        # Read, resize and write the AANet disparity image:
        rgb_disp = cv.imread('DISP_AANET/'+ str(pair[0])+'._pred.png')
        rgb_disp_resized = cv.resize(rgb_disp,(429,323))
        cv.imwrite('CROPPED/AANET/'+ str(pair[0])+ '._pred.png',rgb_disp_resized)

        # Read, crop and write the IR disparity image:
        ir_disp = cv.imread('IR_disp/'+str(pair[1])+'.tiff',cv.IMREAD_UNCHANGED)
        cv.imwrite('CROPPED/IR/'+ str(pair[1])+ '.tiff',ir_disp[98:421,84:513])

#