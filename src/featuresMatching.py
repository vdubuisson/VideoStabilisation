import numpy as np
import cv2
from matplotlib import pyplot as plt

# Defining global variable #
frame_array=[]
global_correction_vector=[]
i=0

# Reading the video for the 1st time, the goal is estimate the overall motion between each frame #
cap = cv2.VideoCapture('../public/bike.avi')

while(cap.isOpened()):
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

                    #if cv2.waitKey(1) & 0xFF == ord('q'):
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
cap2 = cv2.VideoCapture('../public/bike.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('../public/bike_stabilized.avi',fourcc, 10.0, (320,240))
counter=0
buffer_frame=np.empty((240, 320, 3))

while(cap2.isOpened()):
    ret, frame = cap2.read()

    if ret==True and counter<len(global_correction_vector):

        print("Counter:")
        print(counter)
        print("global_correction_vector at this frame:")
        print(global_correction_vector[counter])

        # We iterate through every row of the image displacing every pixels to the right or the left depending on the correction vector value #
        for index, row in enumerate(frame):

            # Two possibilities to manage: The x axis correction is either negative or positive #
            if int(global_correction_vector[counter][0])>0:

                # For now we transform the border pixels into white pixels -> Will be changed to something better #
                if buffer_frame.size:
                    row[:int(global_correction_vector[counter][0])]=buffer_frame[index][:int(global_correction_vector[counter][0])]
                else:
                    row[:int(global_correction_vector[counter][0])]=[255, 255, 255]


                # Displacing every pixels by a certain amount given by the correction vector #
                for index, pix in enumerate(row):
                    if (index>int(global_correction_vector[counter][0])) and (index-int(global_correction_vector[counter][0])<320) :
                        pix=row[index-int(global_correction_vector[counter][0])]

            # Same as before but with the correction vector being negative #
            elif int(global_correction_vector[counter][0])<0:

                if buffer_frame.size:
                    row[int(global_correction_vector[counter][0]):]=buffer_frame[index][int(global_correction_vector[counter][0]):]
                else:
                    row[int(global_correction_vector[counter][0]):]=[255,255,255]

                for index, pix in enumerate(row):
                    if (index>int(global_correction_vector[counter][0])) and (index-int(global_correction_vector[counter][0])<320) :
                        pix=row[index-int(global_correction_vector[counter][0])]

        # The y axis is simple than the x axis because numpy matrix are basically a list of row #
        if int(global_correction_vector[counter][1])>0:
            if buffer_frame.size:
                frame[:int(global_correction_vector[counter][1])]=buffer_frame[:int(global_correction_vector[counter][1])]
            else:
                frame[:int(global_correction_vector[counter][1])]=np.full((320,3), 255, dtype=int)

            for index,row in enumerate(frame):
                if(index>int(global_correction_vector[counter][1])) and (index-int(global_correction_vector[counter][1])<240) :
                    row=frame[index-int(global_correction_vector[counter][1])]


        elif int(global_correction_vector[counter][1])<0:
            if buffer_frame.size:
                frame[int(global_correction_vector[counter][1]):]=buffer_frame[int(global_correction_vector[counter][1]):]
            else:
                frame[int(global_correction_vector[counter][1]):]=np.full((320,3), 255, dtype=int)

            for index,row in enumerate(frame):
                if(index>int(global_correction_vector[counter][1])) and (index-int(global_correction_vector[counter][1])<240) :
                    row=frame[index-int(global_correction_vector[counter][1])]

        # We save the corrected frame #
        out.write(frame)
        buffer_frame=frame
    else:
        # We release the second capture #
        cap2.release()
        out.release()

    counter+=1


# Now we display the initial video and the stabilized one side by side #

cap_initial = cv2.VideoCapture('../public/bike.avi')
cap_stabilized = cv2.VideoCapture('../public/bike_stabilized.avi')

while (cap_initial.isOpened() or cap_stabilized.isOpened()):

    ret_initial, frame_initial=cap_initial.read()
    ret_stabilized, frame_stabilized=cap_stabilized.read()

    if ret_initial:
        cv2.imshow("Initial video:",frame_initial)
    if ret_stabilized:
        cv2.imshow("Stabilized video:",frame_stabilized)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap_initial.release()
cap_stabilized.release()
cv2.destroyAllWindows()
