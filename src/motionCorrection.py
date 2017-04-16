from motionEstimation import *
from motionFiltering import *

def motion_correction (cap, cv2, max_number_frames, windows_cover, window_size, frameWidth, frameHeight):
    global_correction_vector=[]
    global_motion_vector=[]
    frame_array=[]
    i=0
    percentage=0
    print("Starting motion estimation process...")
    while(cap.isOpened() and i<max_number_frames):
        ret, frame = cap.read()
        if ret==True:

            # We stock the frames in an array to access them 2 by 2 #
            frame_array.append(frame)
            if i!=0:


                motion_vector=motion_estimation(frame_array,cv2,i, frameWidth, frameHeight)


                # Here we gather all the motion vector not corrected (just an development process, in the end the correction algorithm will be unified) #
                global_motion_vector.append(motion_vector)

                # Here we compute the correction vector based and the mean value of keypoints displacement #
                #correction_vector=compute_correction_vector(motion_vector)

                # We build a list keeping track of the correction vector needed for every pair of frame #
                #global_correction_vector.append(correction_vector)


            i+=1
            if (i/max_number_frames)>(percentage+0.1):
                percentage+=0.1
                print("{}%  done".format(int(round(percentage*100))))
        else:
            i+=1
            break

    print("Motion estimation process complete!")
    # Here we apply the filtering to all motion vector to eliminate high frequency motion (FPS technique)
    print("Starting motion filtering process...")
    global_correction_vector=motion_filtering(global_motion_vector, windows_cover, window_size)

    #for mv in global_motion_vector:
    #    correction_vector=compute_correction_vector(mv)
    #    global_correction_vector.append(correction_vector)

    #global_correction_vector=global_motion_vector

    return global_correction_vector


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
