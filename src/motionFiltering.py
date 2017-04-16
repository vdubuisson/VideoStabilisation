import numpy as np
import cv2
import copy
import scipy.ndimage
import math
from matplotlib import pyplot as plt

def motion_filtering(global_motion_vector, windows_cover, window_size):

    # Here we create an empty output which will become global correction vector #
    global_correction_vector=np.zeros([len(global_motion_vector), 2])
    global_correction_vector[0][0]=0.0
    global_correction_vector[0][1]=0.0

    # Transformation array is representing the mean displacement of the keypoints for each frame pair#
    transformation_array=np.zeros([len(global_motion_vector), 2])
    i=1

    print("Computing transformation array...")
    for frame_pair in global_motion_vector:

        if frame_pair is not None and i<len(global_motion_vector)-2:
            for kp_displacement in frame_pair:
                transformation_array[i][0]+=kp_displacement[0]
                transformation_array[i][1]+=kp_displacement[1]

            transformation_array[i][0]=transformation_array[i][0]/len(frame_pair)
            transformation_array[i][1]=transformation_array[i][1]/len(frame_pair)

        i+=1

    # After computing transformation_array we display it #
    plt.plot(transformation_array)
    plt.title('Transformation')
    plt.xlabel('Frame number')
    plt.ylabel('Variation in pixels')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.savefig('../public/transformation_array.png')
    plt.show()
    plt.close()

    # The trajectory represent the absolute position of each frame with regards to frame 0 #
    trajectory_array=np.zeros([len(transformation_array), 2])
    trajectory_array[0][0]=0.0
    trajectory_array[0][1]=0.0



    filtered_trajectory=np.zeros(trajectory_array.shape)
    print("Computing trajectory array...")
    j=1
    while j<len(transformation_array):
        trajectory_array[j][0]=trajectory_array[j-1][0]+transformation_array[j][0]
        trajectory_array[j][1]=trajectory_array[j-1][1]+transformation_array[j][1]
        j+=1

    # After computing the trajectory we display it #
    plt.plot(trajectory_array)
    plt.title('Trajectory')
    plt.xlabel('Frame number')
    plt.ylabel('Variation in pixels')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.savefig('../public/trajectory_array.png')
    plt.show()
    plt.close()

    # We resize the trajectory_array to the optimal size value for DFT computation #
    #optimal_size=cv2.getOptimalDFTSize(len(trajectory_array))
    #trajectory_array=np.resize(trajectory_array,(optimal_size,2))

    # Here we compute the number of windows we will apply the DFT onto #
    nb_windows=int(len(trajectory_array)/window_size)+1
    index_windows=0

    mean_trajectory_array=np.zeros((nb_windows, 2))
    print("Filtering the trajectory array...")
    percentage=0

    # Here we manage 2 cases :Windows to overlay on each other or not #
    if windows_cover==1:

        while index_windows<nb_windows:

            # Defining the overlay#
            if index_windows==0:
                start=0
            else:
                start=(index_windows*window_size)-(window_size/2)

            if start+window_size>len(trajectory_array)-1:
                end=len(trajectory_array)-1
            else:
                end=start+window_size

            # Applying the DFT on the x and y axis of the window#
            windowed_trajectory_dft_array_x=np.zeros(trajectory_array[start:end, [0]].shape)
            windowed_trajectory_dft_array_y=np.zeros(trajectory_array[start:end, [1]].shape)

            buffer_frame=np.zeros(trajectory_array.shape)
            if index_windows!=0:
                buffer_frame[start:start+window_size/2]=filtered_trajectory[start:start+window_size/2]
                buffer_frame[start+window_size/2:end]=trajectory_array[start+window_size/2:end]
                cv2.dft(buffer_frame[start:end,[0]], windowed_trajectory_dft_array_x, cv2.DFT_SCALE)
                cv2.dft(buffer_frame[start:end,[1]], windowed_trajectory_dft_array_y, cv2.DFT_SCALE)
            else:
                cv2.dft(trajectory_array[start:end,[0]], windowed_trajectory_dft_array_x, cv2.DFT_SCALE)
                cv2.dft(trajectory_array[start:end,[1]], windowed_trajectory_dft_array_y, cv2.DFT_SCALE)


            # We apply a gaussian filter on the frequency representation of the windowed trajectory => This is a low pass filter#
            windowed_trajectory_dft_array_x=scipy.ndimage.filters.gaussian_filter(windowed_trajectory_dft_array_x,6,0)
            windowed_trajectory_dft_array_y=scipy.ndimage.filters.gaussian_filter(windowed_trajectory_dft_array_y,6,0)

            windowed_filtered_trajectory=np.zeros(trajectory_array[start:end].shape)
            windowed_filtered_trajectory_x=np.zeros(windowed_trajectory_dft_array_x.shape)
            windowed_filtered_trajectory_y=np.zeros(windowed_trajectory_dft_array_y.shape)

            # We apply the IDFT to go back to space representation#
            cv2.idft(windowed_trajectory_dft_array_x, windowed_filtered_trajectory_x, cv2.DFT_SCALE)
            cv2.idft(windowed_trajectory_dft_array_y, windowed_filtered_trajectory_y, cv2.DFT_SCALE)

            windowed_filtered_trajectory[:,[0]]=windowed_filtered_trajectory_x
            windowed_filtered_trajectory[:,[1]]=windowed_filtered_trajectory_y

            # For each window we add to the global filtered_trajectory array#
            filtered_trajectory[start:end]=windowed_filtered_trajectory
            index_windows+=1

    # Same but without windows overlay #
    else:
        while index_windows<nb_windows:

            start=index_windows*window_size

            if start+window_size>len(trajectory_array)-1:
                end=len(trajectory_array)-1
            else:
                end=start+window_size

            mean_counter=start
            mean_trajectory_value=0
            while mean_counter<=end:
                mean_trajectory_value+=trajectory_array[mean_counter]
                mean_counter+=1
            mean_trajectory_array[index_windows]=mean_trajectory_value

            windowed_trajectory_dft_array_x=np.zeros(trajectory_array[start:end, [0]].shape)
            windowed_trajectory_dft_array_y=np.zeros(trajectory_array[start:end, [1]].shape)

            cv2.dft(trajectory_array[start:end,[0]], windowed_trajectory_dft_array_x, cv2.DFT_SCALE)
            cv2.dft(trajectory_array[start:end,[1]], windowed_trajectory_dft_array_y, cv2.DFT_SCALE)

            windowed_trajectory_dft_array_x=scipy.ndimage.filters.gaussian_filter(windowed_trajectory_dft_array_x,0.001,0)
            windowed_trajectory_dft_array_y=scipy.ndimage.filters.gaussian_filter(windowed_trajectory_dft_array_y,0.001,0)

            windowed_filtered_trajectory=np.zeros(trajectory_array[start:end].shape)
            windowed_filtered_trajectory_x=np.zeros(windowed_trajectory_dft_array_x.shape)
            windowed_filtered_trajectory_y=np.zeros(windowed_trajectory_dft_array_y.shape)


            cv2.idft(windowed_trajectory_dft_array_x, windowed_filtered_trajectory_x, cv2.DFT_SCALE)
            cv2.idft(windowed_trajectory_dft_array_y, windowed_filtered_trajectory_y, cv2.DFT_SCALE)

            windowed_filtered_trajectory[:,[0]]=windowed_filtered_trajectory_x
            windowed_filtered_trajectory[:,[1]]=windowed_filtered_trajectory_y

            filtered_trajectory[start:end]=windowed_filtered_trajectory
            index_windows+=1


    # Here we plot the filtered trajectory#
    plt.plot(filtered_trajectory)
    plt.title('Trajectoire after filtering')
    plt.xlabel('Frame number')
    plt.ylabel('Variation in pixels')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.savefig('../public/filtered_trajectory.png')
    plt.show()
    plt.close()

    # Here we filter the trajectory gain but in spatial representation#
    #filtered_trajectory[:,[0]]=scipy.ndimage.filters.gaussian_filter(filtered_trajectory[:,[0]],6,0)
    #filtered_trajectory[:,[1]]=scipy.ndimage.filters.gaussian_filter(filtered_trajectory[:,[1]],6,0)


    plt.plot(filtered_trajectory)
    plt.title('Trajectory after filtering')
    plt.xlabel('Frame number')
    plt.ylabel('Variation in pixels')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.savefig('../public/filtered_trajectory.png')
    plt.show()
    plt.close()

    # Here we compute the global correction vector #

    print("Computing global correction vector...")
    for index,value in enumerate(global_correction_vector):
        if index>0 and index<len(global_correction_vector):
            global_correction_vector[index][0]=filtered_trajectory[index][0]-trajectory_array[index][0]
            global_correction_vector[index][1]=filtered_trajectory[index][1]-trajectory_array[index][1]


    plt.plot(global_correction_vector)
    plt.title('Global correction vector')
    plt.xlabel('Frame number')
    plt.ylabel('Variation in pixels')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.savefig('../public/global_correction_vector.png')
    plt.show()
    plt.close()

    return global_correction_vector
