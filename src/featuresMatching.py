import numpy as np
import cv2
import sys

from matplotlib import pyplot as plt

# Defining global variable #
frame_array=[]
global_correction_vector=[]
i=0

# Border management choice detection #
if len(sys.argv)<3:
    border_type="black"
else:
    border_type=sys.argv[2]

# Here we extract the input file path, the extension and create the output file path #
path=sys.argv[1]
input_extension="."+path.split('.')[len(path.split('.'))-1]
path_stabilized=""

portion_indice=0
for portion in path.split('.')[0:len(path.split('.'))-1]:
    if not portion:
        portion="."
    if portion_indice== len(path.split('.'))-2:
        portion=portion+"_stabilized"
    path_stabilized+=portion
    portion_indice+=1

path_stabilized=path_stabilized+input_extension

# Reading the video for the 1st time, the goal is estimate the overall motion between each frame #
cap = cv2.VideoCapture(path)

# We gather information like frame width, height, codec and FPS rate #
videoFps=int(cap.get(cv2.CAP_PROP_FPS))
frameWidth=int(cap.get(3))
frameHeight=int(cap.get(4))
fourcc=int(cap.get(6))


while(cap.isOpened() and i<300):
    ret, frame = cap.read()
    if ret==True:

        # We stock the frames in an array to access them 2 by 2 #
        frame_array.append(frame)
        if i!=0:

            print("Frame number:")
            print(i)

            # motion_vector containes the displacement of all the keypoints for the current pair of frame #
            motion_vector=[]
            img1=frame_array[i-1]
            img2=frame_array[i]

            # Initiate SIFT detector
            orb = cv2.ORB_create()

            # find the keypoints and descriptors with SIFT and BRIEF
            kp1, des1 = orb.detectAndCompute(img1,None)
            kp2, des2 = orb.detectAndCompute(img2,None)

            if  kp2 and kp1:
                # create BFMatcher object
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

                # Match descriptors.
                matches = bf.match(des1,des2)

                # Sort them in the order of their distance.
                matches = sorted(matches, key = lambda x:x.distance)

                # We iterate through every match to compute the displacement of every keypoints #
                for index, match in enumerate(matches):

                    motion_vector.append((kp2[match.trainIdx].pt[0]-kp1[match.queryIdx].pt[0], kp2[match.trainIdx].pt[1]- kp1[match.queryIdx].pt[1]))

                    # This piece of code is if you want to vizualise the keypoints (useful to debug) #

                    #img_circle1=img1
                    #cv2.circle(img_circle1, (int(kp1[match.queryIdx].pt[0]),int(kp1[match.queryIdx].pt[1])), 3, (0,0,255), 1, 8 )

                    #img_circle2=img2
                    #cv2.circle(img_circle2, (int(kp2[match.trainIdx].pt[0]),int(kp2[match.trainIdx].pt[1])), 3, (0,0,255), 1, 8 )

                    #cv2.imshow("img_circle1",img_circle1)
                    #cv2.imshow("img_circle2",img_circle2)

                    #cv2.imshow("test", frame)
                    #if cv2.waitKey(20):
                    #    break



                ####### TO DO : FILTRATE THE DISPLACEMENT TO ONLY KEEEP THE LOW FREQUENCY WITH FPS ########
                #
                #
                #
                ####### TO DO : FILTRATE THE DISPLACEMENT TO ONLY KEEEP THE LOW FREQUENCY WITH FPS ########


                # Here we compute the correction vector based and the mean value of keypoints displacement #

                correction_vector_x=0
                correction_vector_y=0

                for item in motion_vector:
                    correction_vector_x+=item[0]
                    correction_vector_y+=item[1]

                correction_vector_x=correction_vector_x/len(motion_vector)
                correction_vector_y=correction_vector_y/len(motion_vector)

                correction_vector=(correction_vector_x, correction_vector_y)
                print("correction_vector:")
                print(correction_vector)

                # We build a list keeping track of the correction vector needed for every pair of frame #
                global_correction_vector.append(correction_vector)

        i+=1
    else:
        i+=1
        break


print("global_correction_vector at the end:")
print(global_correction_vector)
cap.release()
cv2.destroyAllWindows()

####### TO DO : MANAGE BORDERS TO GET A CONSTANT SIZE IMAGE ########
#
#
#
####### TO DO : MANAGE BORDERS TO GET A CONSTANT SIZE IMAGE ########


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
