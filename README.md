# Rover-Project-One
Rover Project

The goal of this project writeup is to demonstrate the code used to solve the rover project:

I downloaded the simulator and recorded data in "Training Mode"
I tested out the functions in the Jupyter Notebook provided
I added functions to detect obstacles and samples.
I demonstrated my mapping pipeline by filling in the process_image() function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw image to a map. 
I used moviepy to process the images in my saved datadet with the process_image() function. (See video below)

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
#%matplotlib qt # Choose %matplotlib qt to plot to an interactive window (note it may show up behind your browser)
# Make some of the relevant imports
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
# Grab a random image and display it
idx = np.random.randint(0, len(img_list)-1)
image = mpimg.imread(img_list[idx])
plt.imshow(image)
# Autonomous Navigation and Mapping
```
Running my image data

```
path = '../Python Programs/IMG/*'
img_list = glob.glob(path)
# Grab a random image and display it
idx = np.random.randint(0, len(img_list)-1)
image = mpimg.imread(img_list[idx])
plt.imshow(image)
```

```
<matplotlib.image.AxesImage at 0x7f95c4cc54a8>
```
Running test calibration data

```
# In the simulator you can toggle on a grid on the ground for calibration
# You can also toggle on the rock samples with the 0 (zero) key.  
# Here's an example of the grid and one of the rocks
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

Running my calibration data

```
# In the simulator you can toggle on a grid on the ground for calibration
# You can also toggle on the rock samples with the 0 (zero) key.  
# Here's an example of the grid and one of the rocks
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
# Define a function to perform a perspective transform
# I've used the example grid image above to choose source points for the
# grid cell in front of the rover (each grid cell is 1 square meter in the sim)
# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Define calibration box in source (actual) and destination (desired) coordinates
# These source and destination points are defined to warp the image
# to a grid where each 10x10 pixel square represents 1 square meter
# The destination box will be 2*dst_size on each side
dst_size = 5 
# Set a bottom offset to account for the fact that the bottom of the image 
# is not the position of the rover but a bit in front of it
# this is just a rough guess, feel free to change it!
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
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
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

# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # TODO:
    # Convert yaw to radians
    # Apply a rotation
    xpix_rotated = 0
    ypix_rotated = 0
    # Return the result  
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # TODO:
    # Apply a scaling and a translation
    xpix_translated = 0
    ypix_translated = 0
    # Return the result  
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
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

# Grab another random image
idx = np.random.randint(0, len(img_list)-1)
image = mpimg.imread(img_list[idx])
warped = perspect_transform(image, source, destination)
threshed = color_thresh(warped)

# Calculate pixel values in rover-centric coords and distance/angle to all pixels
xpix, ypix = rover_coords(threshed)
dist, angles = to_polar_coords(xpix, ypix)
mean_dir = np.mean(angles)

# Do some plotting
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

I filled in the perception_step() function within the perception.py script with the appropriate image processing functions
to create a map and update Rover() data. I filled in the decision_step() function within the perception.py script with conditional statements. I iterated on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating 
and mapping.

```
# Import pandas and read in csv file as a dataframe
import pandas as pd
# Change this path to your data directory
df = pd.read_csv('./test_dataset/robot_log.csv')
csv_img_list = df["Path"].tolist() # Create list of image pathnames
# Read in ground truth map and create a 3-channel image with it
ground_truth = mpimg.imread('./calibration_images/map_bw.png')
ground_truth_3d = np.dstack((ground_truth*0, ground_truth*255, ground_truth*0)).astype(np.float)

# Creating a class to be the data container
# Will read in saved data from csv file and populate this object
# Worldmap is instantiated as 200 x 200 grids corresponding 
# to a 200m x 200m space (same size as the ground truth map: 200 x 200 pixels)
# This encompasses the full range of output position values in x and y from the sim
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

# Instantiate a Databucket().. this will be a global variable/object
# that you can refer to in the process_image() function below
data = Databucket()
```
```
# Define a function to pass stored images to
# reading rover position and yaw angle from csv file
# This function will be used by moviepy to create an output video
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
        # First create a blank image (can be whatever shape you like)
    output_image = np.zeros((img.shape[0] + data.worldmap.shape[0], img.shape[1]*2, 3))
        # Next you can populate regions of the image with various output
        # Here I'm putting the original image in the upper left hand corner
    output_image[0:img.shape[0], 0:img.shape[1]] = img

        # Let's create more images to add to the mosaic, first a warped image
    warped = perspect_transform(img, source, destination)
        # Add the warped image in the upper right hand corner
    output_image[0:img.shape[0], img.shape[1]:] = warped

        # Overlay worldmap with ground truth map
    map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.5, 0)
        # Flip map overlay so y-axis points upward and add to output_image 
    output_image[img.shape[0]:, 0:data.worldmap.shape[1]] = np.flipud(map_add)


        # Then putting some text over the image
    cv2.putText(output_image,"Populate this image with your analyses to make a video!", (20, 20), 
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    data.count += 1 # Keep track of the index in the Databucket()
    
    return output_image
    ```
    

Autonomous Navigation and Mapping

Perception

The first change was to again add the rgb_thresh_max to the color_thresh function. In my perception_step function, I set
the Rover.vision_image red channel to rock_sample color threshold output, green channel for walls, and blue channel for
navigable terrain. I the converted all three of the thresholded outputs into rover-centric coordinates. The navigable terrain
and wall coordinates were transformed to world coordinates, and applied as follows:

Blue channel of worldmap received += 255 for all navigable terrain and -=255 for walls

Red channel of worldmap receives +=255 for all walls and -=255 for navigable terrain

i found that this method gave me the highest fidelity. As for the rover-centric rock sample coordinates, they were used to
check if there is a sample in the image, and determine the angle the rover needs to turn to get to that sample. If a 
sample was in sight, the rovers mode was set to 'rock_visible', a custom mode I handle in the decision_step. Further, if
no rock sample is present, the navigable terrain rover-centric coordinates are converned to polor coordinates, and set to 
the rovers nav_dists and nav+angles variable.

Decision

In the decision_step function, I added a conditional check if the rover is currently picking up a sample, and if so to stop 
any movemoent and set the mode to 'stop'.
The last conditional I added was a check if the rover mode was 'rock_visible'. If it was, then max out the throttle and steer toward the rock, regardless if there is not enough navigable terrain to continue much farther. This was done because the rock samples are close to the walls, and the normal 'forward' mode tells the robot to stop and turn away from walls.

2. Launching in autuonomous mode your rover can navigate and map autuonomously. Explain your results and how you might improve them in 
your writeup.

Note: running the simulator with different choices of resolution and graphics quality may produce different results,
particularly on didfferent machines! Make a note of your simulator settings (resolution and rgaphics quality set on launch) and frames per second (FPS output to terminal by drive_rover.py ) in your writeup when you submit the project so your reviewer can reproduce your results.

My simulator was running at ~30 FPS with 1280x768 resolution and 'Beautiful' graphics quality.

My approach is working for the most part, and is able to map a good percentage of the environment it covers with ~85 % fidelity. I wasn't able to test the simulator for longer than a couple minutes due to a memory leak in the simulator. My rover can also successfully see and pickup rock samples, but it does not yet return them to its original position. I could have included that the 'easy' way, waiting for the robot to randomly enter its starting position, but I would like to implement it in a more correct way in the future.

A few drawbacks of my approach include that there is no intelligent algorithm for exploration of the map, so the rover will cover the same spots several times, and miss some spots where the mean angle does not take it. There is also nothing to prevent the rover from driving in a circle for a while when following the mean angle of navigable terrain. The last caveat of my rover is that in the case of getting stuck, there is no 'get unstuck' routine.

I would like to revisit this project in the future, and try some end to end deep learning for this rover. I would also like to fix my current approach in the near future by making my rover a wall crawler. This project was a lot of fun!


Perception
The perception system was a pipeline of image processing steps performed on an input image from the simulator.
Here is an example of two images from the rovers camera, one of which contains a rock sample:
The first processing step was a prespective transform to get a top-down view of the input image.
Next I needed to apply a color threshold in order to get a gray-scale image. I used several colo thresholds in order
to separate walls, navigable terrian, and rock samples.
The last step was to take these (x,y) points and convert them to rover centric coordinates. To do this, we move
the coordinate system to make rovers location be (0,0).
THis red arrow is showing the mean angle. if we represent all in polar coordinates (distance, angle). The mean angle
was the method used to steer the rover.
An extra step in the perception pipeline was converting these navigable terrain or wall points into world coordinates,
in order to map the environment. Becuase we had available the yaw the rover, we were able to perform a rotation to
match the x axis of the rover and world map, along with a translation and scaling function to get the correct world
map, along with a translation and scaling function to get the correct world coordinates.

Descision Making
Broken down, the rover needed to decide every fram what values to set for the following controls:
Steering angle
Throttle value
Brake value

To make the these decisions, a few more things had to be considered. The steering angle came from the mean angle you
saw above, but what if you are headed straight for a wall? Unless you are going very slow, you won't be able to 
simply turn to escape without braking.

This is an example of a decision making problem with an infinite amount of soultions. In my case, I chose to solve it 
by first stopping the rover if the amount of navigable terrain points didn't meet my thrshold, and second to
turn in place in the direction of the greater amount of terrain points.

There were several more cases I handled in my code like the the above, but they were all along the lines of 'if this', 
'do that'.

As this was the first project, we weren't required to calculate any kinematics for our rover. The action step mainly 
consisted of sending instructions to the simulator, where the rover would read and execute them.

Ideas and Review
I think a lot of students including myself will not be 'done' with this project for a while. There are many more things to 
try! Some idead I would still like to test include the wall crawler method and using reinforcement learning to teach the 
rover.

Being in the first cohort of the Robotics ND, it is our job to give feedback on the program to improve it for future
students. From my perspective, this project covered the basics of modern robotics quite well, and the simulated rover
was a lot of fun to work with. There were some bugs and memory leaks in the simulator, but the Udacity Robotics team 
were on top of them quickly. Future cohorts should have an even smoother ride with their rover:)


