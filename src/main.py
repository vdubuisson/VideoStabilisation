import numpy as np
import cv2
import sys
import getopt

from matplotlib import pyplot as plt
from utils import *
from motionEstimation import *
from motionCorrection import *
from motionFiltering import *
from stabilizer import *

print("Intializing...")
# Managing arguments #
max_number_frames=0
windows_cover=0
window_size=60
border_type="black"
path=""


args = '-a -b -cfoo -d bar a1 a2'.split()

try:
    opts, args = getopt.getopt(sys.argv[1:],"",["ifile=","border=", "max_frames=", "windows_cover=", "window_size="])
except getopt.GetoptError:
    print ('main.py --ifile <inputfile> --border <type of border> --max_frames <integer> --windows_cover <integer> --window_size <integer>')
    sys.exit(2)



for opt, arg in opts:
    if opt == '--ifile':
        path=arg
    elif opt == '--border':
        border_type=arg
    elif opt== '--max_frames':
        max_number_frames=int(arg)
    elif opt=='--windows_cover':
        windows_cover=int(arg)
    elif opt=='--window_size':
        window_size=int(arg)



# Here we extract the input file path, the extension and create the output file path #
#path=sys.argv[1]
input_extension="."+path.split('.')[len(path.split('.'))-1]
path_stabilized=create_stabilized_path(path, input_extension)

print("Reading input video...")
# Reading the video for the 1st time, the goal is estimate the overall motion between each frame #
cap = cv2.VideoCapture(path)

# We gather information like frame width, height, codec and FPS rate #
videoFps=int(cap.get(cv2.CAP_PROP_FPS))
frameWidth=int(cap.get(3))
frameHeight=int(cap.get(4))
fourcc=int(cap.get(6))
numberOfFrames=int(cap.get(7))

# Here we extract the number of frame we want to limit ourselves to. If the user did not precise any value we read the whole video #

#max_number_frames=max_number_frames(sys.argv, numberOfFrames)
if max_number_frames==0 or max_number_frames>numberOfFrames:
    max_number_frames=numberOfFrames

global_correction_vector=motion_correction(cap, cv2, max_number_frames, windows_cover, window_size, frameWidth, frameHeight)

cap.release()
cv2.destroyAllWindows()

# We now read the video a second time while applying the correction to see the result #
cap2 = cv2.VideoCapture(path)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(path_stabilized,fourcc, videoFps, (frameWidth,frameHeight))
buffer_frame=np.empty((frameHeight, frameWidth, 3))

print("Creating a stabilized version of the input video...")
image_warping(cap2, fourcc, out, buffer_frame, max_number_frames, global_correction_vector, np, cv2, frameWidth, frameHeight, border_type)
print("Done!!!")
# Now we display the initial video and the stabilized one side by side #

cap_initial = cv2.VideoCapture(path)
cap_stabilized = cv2.VideoCapture(path_stabilized)

display_two_vids(cap_initial, cap_stabilized, max_number_frames, videoFps)

cap_initial.release()
cap_stabilized.release()
cv2.destroyAllWindows()

print("Stabilized video can be found at : {}".format(path_stabilized))
