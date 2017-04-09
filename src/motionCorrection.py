from motionEstimation import *
from motionFiltering import *

def motion_correction (cap, cv2, max_number_frames, windows_cover, window_size):
    global_correction_vector=[]
    global_motion_vector=[]
    frame_array=[]
    i=0
    while(cap.isOpened() and i<max_number_frames):
        ret, frame = cap.read()
        if ret==True:

            # We stock the frames in an array to access them 2 by 2 #
            frame_array.append(frame)
            if i!=0:

                print("Frame number:")
                print(i)

                motion_vector=motion_estimation(frame_array,cv2,i)


                # Here we gather all the motion vector not corrected (just an development process, in the end the correction algorithm will be unified) #
                global_motion_vector.append(motion_vector)

                # Here we compute the correction vector based and the mean value of keypoints displacement #
                #correction_vector=compute_correction_vector(motion_vector)

                # We build a list keeping track of the correction vector needed for every pair of frame #
                #global_correction_vector.append(correction_vector)


            i+=1
        else:
            i+=1
            break

    # Here we apply the filtering to all motion vector to eliminate high frequency motion (FPS technique)
    global_correction_vector=motion_filtering(global_motion_vector, windows_cover, window_size)

    #for mv in global_motion_vector:
    #    correction_vector=compute_correction_vector(mv)
    #    global_correction_vector.append(correction_vector)

    #global_correction_vector=global_motion_vector

    return global_correction_vector

def draw_circle_keypoints(cv2, img1, img2, kp1, kp2, match, frame):
    # This piece of code is if you want to vizualise the keypoints (useful to debug) #

    img_circle1=img1
    cv2.circle(img_circle1, (int(kp1[match.queryIdx].pt[0]),int(kp1[match.queryIdx].pt[1])), 3, (0,0,255), 1, 8 )

    img_circle2=img2
    cv2.circle(img_circle2, (int(kp2[match.trainIdx].pt[0]),int(kp2[match.trainIdx].pt[1])), 3, (0,0,255), 1, 8 )

    cv2.imshow("img_circle1",img_circle1)
    cv2.imshow("img_circle2",img_circle2)

    cv2.imshow("test", frame)


def compute_correction_vector(motion_vector):

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

    return correction_vector
