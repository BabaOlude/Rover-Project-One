# Rover-Project-One
Rover Project

The goal of this project writeup is to demonstrate the code used to solve the rover project:

I downloaded the simulator and recorded data in "Training Mode"
I tested out the functions in the Jupyter Notebook provided
I added functions to detect obstacles and samples.
I demonstrated my mapping pipeline by filling in the process_image() function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw image to a map. 
I used moviepy to process the images in my saved datadet with the process_image() function. (See video below)

# Notebook Analysis

1. I ran the functions provided in the notebook on test images with the test data provided and with the test data I have recorded. I modified the functions to allow for color selection of obstacles.

I selected two images from training data I recorded, one for testing the color filter on ground, and the other for a rock
sample.

Here are the above images after the perspective transform to top down view

I modified the color_threshold function to include a rgb_thresh_max parameter, to set maximum values for the RGB thresholds in addition to the minimums. Here is the output on the same training images



2. I populated the process_image() function with the appropriate analysis steps to map pixels identifying navigable
terrain, obstacles and rock samples into a worldmap. I also ran process_image() on my test data using the moviepy functions
provided to create video output of my results.

# Autonomous Navigation and Mapping

Autonomous Navigation / Mapping

Fill in the perception_step() function within the perception.py script with the appropriate image processing functions
to create a map and update Rover() data (similar to what you did process_image() in the notebook).
Fill in the decision_step() function within the perception.py script with conditional statements that take into consideration the 
outputs of the perception_step() in deciding how to issue throttle, brake and steering commands.
Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating 
and mapping.

Rubirc Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

Writeup/README
1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as
markdown or pdf.

You're reading it!



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


