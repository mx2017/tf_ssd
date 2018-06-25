import numpy as np
import cv2

img_file = '000001.png'
anchor = np.array(
      [[  36.,  37.], [ 366., 174.], [ 115.,  59.],
       [ 162.,  87.], [  38.,  90.], [ 258., 173.],
       [ 224., 108.], [  78., 170.], [  72.,  43.]])

img = cv2.imread(img_file)
#cv2.rectangle(img, (384,0), (510, 128), (0, 255, 0), 3)  
rows,cols,channels = img.shape
color_B = 50
color_G = 255
for i in range(0,len(anchor)):
    center_x = int(cols/2)
    center_y = int(rows/2)
    start_x = center_x - int(anchor[i][0]/2)
    start_y = center_y - int(anchor[i][1]/2)
    end_x = start_x + int(anchor[i][0])
    end_y = start_y + int(anchor[i][1])
    color_G -= 20
    color_B += 20
    cv2.rectangle(img, (start_x,start_y), (end_x, end_y), (color_B, color_G, 50), 3) 

cv2.imwrite("test.png",img)
#cv2.imshow("show",img)

#cv2.waitKey()


       
       
