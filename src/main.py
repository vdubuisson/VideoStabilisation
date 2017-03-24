import numpy as np
import cv2
import sys

from matplotlib import pyplot as plt
from utils import *
from motionEstimation import *

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

global_correction_vector=motion_correction(cap, cv2)

print("global_correction_vector at the end:")
print(global_correction_vector)
cap.release()
cv2.destroyAllWindows()

# We now read the video a second time while applying the correction to see the result #
cap2 = cv2.VideoCapture(path)
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter(path_stabilized,fourcc, videoFps, (frameWidth,frameHeight))
counter=0
buffer_frame=np.empty((frameHeight, frameWidth, 3))

while(cap2.isOpened() and counter<300):
    ret, frame = cap2.read()

    if ret==True and counter<len(global_correction_vector):

        print("Counter:")
        print(counter)
        print("global_correction_vector at this frame:")
        print(global_correction_vector[counter])

        shiftX = int(global_correction_vector[counter][0])
        shiftY = int(global_correction_vector[counter][1])

        # We create a numpy matrix to shift the frame according to the correction vector #
        shiftMatrix = np.float32([[1, 0, shiftX], [0, 1, shiftY]])

        # We shift the frame with the matrix #
        newFrame = cv2.warpAffine(frame, shiftMatrix, (frameWidth, frameHeight))

        # Borders a managed by using previous frame pixels #
        if border_type=="replace":
            if shiftX > 0 :
                # Left border #
                newFrame[0:frameHeight, 0:shiftX] = buffer_frame[0:frameHeight, 0:shiftX]
                newFrame[0:frameHeight, 0:shiftX+2] = cv2.blur(newFrame[0:frameHeight, 0:shiftX+2], (5,5))
            elif shiftX < 0 :
                # Right border #
                newFrame[0:frameHeight, frameWidth+shiftX:frameWidth] = buffer_frame[0:frameHeight, frameWidth+shiftX:frameWidth]
                newFrame[0:frameHeight, frameWidth+shiftX-2:frameWidth] = cv2.blur(newFrame[0:frameHeight, frameWidth+shiftX-2:frameWidth], (5,5))
            if shiftY > 0 :
                # Top border #
                newFrame[0:shiftY, 0:frameWidth] = buffer_frame[0:shiftY, 0:frameWidth]
                newFrame[0:shiftY+2, 0:frameWidth] = cv2.blur(newFrame[0:shiftY+2, 0:frameWidth], (5,5))
            elif shiftY < 0 :
                # Bottom border #
                newFrame[frameHeight+shiftY:frameHeight, 0:frameWidth] = buffer_frame[frameHeight+shiftY:frameHeight, 0:frameWidth]
                newFrame[frameHeight+shiftY-2:frameHeight, 0:frameWidth] = cv2.blur(newFrame[frameHeight+shiftY-2:frameHeight, 0:frameWidth], (5,5))

        # Border are filled with white pixels #
        elif border_type=="white":
            if shiftX > 0 :
                # Left border #
                newFrame[0:frameHeight, 0:shiftX] = np.full((frameHeight, frameWidth, 3), 255, dtype=int)[0:frameHeight, 0:shiftX]
                #newFrame[0:frameHeight, 0:shiftX+2] = cv2.blur(newFrame[0:frameHeight, 0:shiftX+2], (5,5))
            elif shiftX < 0 :
                # Right border #
                newFrame[0:frameHeight, frameWidth+shiftX:frameWidth] = np.full((frameHeight, frameWidth, 3), 255, dtype=int)[0:frameHeight, frameWidth+shiftX:frameWidth]
                #newFrame[0:frameHeight, frameWidth+shiftX-2:frameWidth] = cv2.blur(newFrame[0:frameHeight, frameWidth+shiftX-2:frameWidth], (5,5))
            if shiftY > 0 :
                # Top border #
                newFrame[0:shiftY, 0:frameWidth] = np.full((frameHeight, frameWidth, 3), 255, dtype=int)[0:shiftY, 0:frameWidth]
                #newFrame[0:shiftY+2, 0:frameWidth] = cv2.blur(newFrame[0:shiftY+2, 0:frameWidth], (5,5))
            elif shiftY < 0 :
                # Bottom border #
                newFrame[frameHeight+shiftY:frameHeight, 0:frameWidth] = np.full((frameHeight, frameWidth, 3), 255, dtype=int)[frameHeight+shiftY:frameHeight, 0:frameWidth]
                #newFrame[frameHeight+shiftY-2:frameHeight, 0:frameWidth] = cv2.blur(newFrame[frameHeight+shiftY-2:frameHeight, 0:frameWidth], (5,5))

        # Border are filled with white pixels #
        elif border_type=="black":
            if shiftX > 0 :
                # Left border #
                newFrame[0:frameHeight, 0:shiftX] = np.full((frameHeight, frameWidth, 3), 0, dtype=int)[0:frameHeight, 0:shiftX]
                #newFrame[0:frameHeight, 0:shiftX+2] = cv2.blur(newFrame[0:frameHeight, 0:shiftX+2], (5,5))
            elif shiftX < 0 :
                # Right border #
                newFrame[0:frameHeight, frameWidth+shiftX:frameWidth] = np.full((frameHeight, frameWidth, 3), 0, dtype=int)[0:frameHeight, frameWidth+shiftX:frameWidth]
                #newFrame[0:frameHeight, frameWidth+shiftX-2:frameWidth] = cv2.blur(newFrame[0:frameHeight, frameWidth+shiftX-2:frameWidth], (5,5))
            if shiftY > 0 :
                # Top border #
                newFrame[0:shiftY, 0:frameWidth] = np.full((frameHeight, frameWidth, 3), 0, dtype=int)[0:shiftY, 0:frameWidth]
                #newFrame[0:shiftY+2, 0:frameWidth] = cv2.blur(newFrame[0:shiftY+2, 0:frameWidth], (5,5))
            elif shiftY < 0 :
                # Bottom border #
                newFrame[frameHeight+shiftY:frameHeight, 0:frameWidth] = np.full((frameHeight, frameWidth, 3), 0, dtype=int)[frameHeight+shiftY:frameHeight, 0:frameWidth]
                #newFrame[frameHeight+shiftY-2:frameHeight, 0:frameWidth] = cv2.blur(newFrame[frameHeight+shiftY-2:frameHeight, 0:frameWidth], (5,5))


        # We apply a blur on the new borders

        # We save the corrected frame #
        out.write(newFrame)
        buffer_frame=newFrame
    else:
        # We release the second capture #
        cap2.release()
        out.release()

    counter+=1


# Now we display the initial video and the stabilized one side by side #

cap_initial = cv2.VideoCapture(path)
cap_stabilized = cv2.VideoCapture(path_stabilized)

j=0
while ((cap_initial.isOpened() or cap_stabilized.isOpened()) and j<300):

    ret_initial, frame_initial=cap_initial.read()
    ret_stabilized, frame_stabilized=cap_stabilized.read()

    if ret_initial:
        cv2.imshow("Initial video:",frame_initial)
    if ret_stabilized:
        cv2.imshow("Stabilized video:",frame_stabilized)
    if cv2.waitKey(int((1/videoFps)*1000)) & 0xFF == ord('q'):
        break
    j+=1

cap_initial.release()
cap_stabilized.release()
cv2.destroyAllWindows()
