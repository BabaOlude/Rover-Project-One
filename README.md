# Rover-Project-One
Rover Project

The goal of this project writeup is to demonstrate the code used to solve the rover project:

I downloaded the simulator and recorded data in "Training Mode".
I tested out the functions in the Jupyter Notebook provided.
I added functions to detect obstacles and samples.
I demonstrated my mapping pipeline by filling in the process_image() function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw image to a map. 
I used moviepy to process the images in my saved dataset with the process_image() function. (See video below)

# Notebook Analysis

I ran the functions provided in the notebook on test images with the test data provided and with the test data I have recorded. I modified the functions to allow for color selection of obstacles.

I selected two images from training data I recorded, one for testing the color filter on ground, and the other for a rock
sample. To detect navigable areas, I used the default color threshold.

I populated the process_image() function with the appropriate analysis steps to map pixels identifying navigable
terrain, obstacles and rock samples into a worldmap. I also ran process_image() on my test data using the moviepy functions
provided to create video output of my results.

```%%HTML
<style> code {background-color : orange !important;} </style>
```
```
%matplotlib inline
#%matplotlib qt # Choose %matplotlib qt 
# imports
import cv2 # OpenCV for perspective transform
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.misc # For saving images as needed
import glob  # For reading in a list of images from a folder
```

Running test image data

```
path = './test_dataset/IMG/*'
img_list = glob.glob(path)

idx = np.random.randint(0, len(img_list)-1)
image = mpimg.imread(img_list[idx])
plt.imshow(image)

```
Running my image data

```
path = '../Python Programs/IMG/*'
img_list = glob.glob(path)

idx = np.random.randint(0, len(img_list)-1)
image = mpimg.imread(img_list[idx])
plt.imshow(image)
```

```
<matplotlib.image.AxesImage at 0x7f95c4cc54a8>
```
Running test calibration data

```

  

example_grid = './calibration_images/example_grid1.jpg'
example_rock = './calibration_images/example_rock1.jpg'
grid_img = mpimg.imread(example_grid)
rock_img = mpimg.imread(example_rock)

fig = plt.figure(figsize=(12,3))
plt.subplot(121)
plt.imshow(grid_img)
plt.subplot(122)
plt.imshow(rock_img)
```

calibration 

```

  

example_grid = '../Python Programs/example_grid1.jpg'
example_rock = '../Python Programs/example_rock1.jpg'
grid_img = mpimg.imread(example_grid)
rock_img = mpimg.imread(example_rock)

fig = plt.figure(figsize=(12,3))
plt.subplot(121)
plt.imshow(grid_img)
plt.subplot(122)
plt.imshow(rock_img)
```

```




def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Define calibration box in source (actual) and destination (desired) coordinates



dst_size = 5 
 


bottom_offset = 6
source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])
warped = perspect_transform(grid_img, source, destination)
plt.imshow(warped)
#scipy.misc.imsave('../output/warped_example.jpg', warped)
```
```# Identify pixels above the threshold
# Threshold of RGB > 160 
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    
    color_select = np.zeros_like(img[:,:,0])
    
    
    
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

threshed = color_thresh(warped)
plt.imshow(threshed, cmap='gray')
#scipy.misc.imsave('../output/warped_threshed.jpg', threshed*255)
```
```
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # TODO:
    # Convert yaw to radians
    # Apply a rotation
    xpix_rotated = 0
    ypix_rotated = 0
    # Return the result  
    return xpix_rotated, ypix_rotated

# perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # TODO:
    # Apply a scaling and a translation
    xpix_translated = 0
    ypix_translated = 0
    # Return the result  
    return xpix_translated, ypix_translated

# Define a function 
# function 
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# random image
idx = np.random.randint(0, len(img_list)-1)
image = mpimg.imread(img_list[idx])
warped = perspect_transform(image, source, destination)
threshed = color_thresh(warped)

# ixels
xpix, ypix = rover_coords(threshed)
dist, angles = to_polar_coords(xpix, ypix)
mean_dir = np.mean(angles)

# plotting
fig = plt.figure(figsize=(12,9))
plt.subplot(221)
plt.imshow(image)
plt.subplot(222)
plt.imshow(warped)
plt.subplot(223)
plt.imshow(threshed, cmap='gray')
plt.subplot(224)
plt.plot(xpix, ypix, '.')
plt.ylim(-160, 160)
plt.xlim(0, 160)
arrow_length = 100
x_arrow = arrow_length * np.cos(mean_dir)
y_arrow = arrow_length * np.sin(mean_dir)
plt.arrow(0, 0, x_arrow, y_arrow, color='red', zorder=2, head_width=10, width=2)
```

# Autonomous Navigation / Mapping

Autonomous Navigation and Mapping

Perception

I filled in the perception_step() function within the perception.py script with the appropriate image processing functions
to create a map and update Rover() data. I filled in the decision_step() function within the perception.py script with conditional statements. I iterated the perception and decision functions until acheiving navigation and mapping.

I the converted all three of the thresholded outputs (red, green, blue) into rover-centric coordinates. The navigable terrain
and wall coordinates were transformed to world coordinates. The basic steps led me to my highest fidelity. Rock sample coordinates were used to check if there is a sample in the image. If a sample was in sight, the rovers mode was set in the decision_step. If no rock sample is present the coordinates stayed in polor coordinates.
```

import pandas as pd

df = pd.read_csv('./test_dataset/robot_log.csv')
csv_img_list = df["Path"].tolist() # Create list of image pathnames

ground_truth = mpimg.imread('./calibration_images/map_bw.png')
ground_truth_3d = np.dstack((ground_truth*0, ground_truth*255, ground_truth*0)).astype(np.float)



 


class Databucket():
    def __init__(self):
        self.images = csv_img_list  
        self.xpos = df["X_Position"].values
        self.ypos = df["Y_Position"].values
        self.yaw = df["Yaw"].values
        self.count = -1 # This will be a running index, setting to -1 is a hack
                        # because moviepy (below) seems to run one extra iteration
        self.worldmap = np.zeros((200, 200, 3)).astype(np.float)
        self.ground_truth = ground_truth_3d # Ground truth worldmap



data = Databucket()
```
```



def process_image(img):
    # Example of how to use the Databucket() object defined above
    # to print the current x, y and yaw values 
    # print(data.xpos[data.count], data.ypos[data.count], data.yaw[data.count])

    # TODO: 
    # 1) Define source and destination points for perspective transform
    # 2) Apply perspective transform
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    # 4) Convert thresholded image pixel values to rover-centric coords
    # 5) Convert rover-centric pixel values to world coords
    # 6) Update worldmap (to be displayed on right side of screen)
        # Example: data.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          data.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          data.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    # 7) Make a mosaic image, below is some example code
        
    output_image = np.zeros((img.shape[0] + data.worldmap.shape[0], img.shape[1]*2, 3))
        
        
    output_image[0:img.shape[0], 0:img.shape[1]] = img

        
    warped = perspect_transform(img, source, destination)
        
    output_image[0:img.shape[0], img.shape[1]:] = warped

        
    map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.5, 0)
         
    output_image[img.shape[0]:, 0:data.worldmap.shape[1]] = np.flipud(map_add)


        
    cv2.putText(output_image,"Populate this image with your analyses to make a video!", (20, 20), 
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    data.count += 1 # Keep track of the index in the Databucket()
    
    return output_image
```

# Decision

In the decision_step function, I added a conditional check if the rover is currently picking up a sample, and if so to stop 
any movement and set the mode to 'stop'.

My approach worked and covers the environment with ~85 % fidelity. My rover successfully see and pickup rock samples. 
