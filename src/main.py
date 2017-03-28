import numpy as np
import cv2
import sys

from matplotlib import pyplot as plt
from utils import *
from motionEstimation import *
from motionCorrection import *
from motionFiltering import *
from stabilizer import *

# Defining global variable #


# Border management choice detection #
border_type=border_management(sys.argv)

# Here we extract the input file path, the extension and create the output file path #
path=sys.argv[1]
input_extension="."+path.split('.')[len(path.split('.'))-1]
path_stabilized=create_stabilized_path(path, input_extension)

# Reading the video for the 1st time, the goal is estimate the overall motion between each frame #
cap = cv2.VideoCapture(path)

# We gather information like frame width, height, codec and FPS rate #
videoFps=int(cap.get(cv2.CAP_PROP_FPS))
frameWidth=int(cap.get(3))
frameHeight=int(cap.get(4))
fourcc=int(cap.get(6))
numberOfFrames=int(cap.get(7))

# Here we extract the number of frame we want to limit ourselves to. If the user did not precise any value we read the whole video #

max_number_frames=max_number_frames(sys.argv, numberOfFrames)

global_correction_vector=motion_correction(cap, cv2, max_number_frames)

print("global_correction_vector at the end:")
print(global_correction_vector)
cap.release()
cv2.destroyAllWindows()

# We now read the video a second time while applying the correction to see the result #
cap2 = cv2.VideoCapture(path)
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter(path_stabilized,fourcc, videoFps, (frameWidth,frameHeight))
buffer_frame=np.empty((frameHeight, frameWidth, 3))

image_warping(cap2, fourcc, out, buffer_frame, max_number_frames, global_correction_vector, np, cv2, frameWidth, frameHeight, border_type)

# Now we display the initial video and the stabilized one side by side #

cap_initial = cv2.VideoCapture(path)
cap_stabilized = cv2.VideoCapture(path_stabilized)

display_two_vids(cap_initial, cap_stabilized, max_number_frames, videoFps)

cap_initial.release()
cap_stabilized.release()
cv2.destroyAllWindows()
