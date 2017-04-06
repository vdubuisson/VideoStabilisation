import numpy as np
import cv2
import copy
import scipy.ndimage
from matplotlib import pyplot as plt

def motion_filtering(global_motion_vector):

    global_correction_vector=np.empty([len(global_motion_vector), 2])
    global_correction_vector[0][0]=0.0
    global_correction_vector[0][1]=0.0
    #filtered_global_motion_vector=copy.deepcopy(global_motion_vector)

    #file_before_filtering=open('../public/file_before_filtering.txt', 'w')
    #file_after_filtering=open('../public/file_after_filtering.txt', 'w')

    #for item in global_motion_vector:
    #    file_before_filtering.write("%s\n" % item)

    transformation_array=np.empty([len(global_motion_vector), 2])
    i=1
    transformation_array[0][0]=0.0
    transformation_array[0][1]=0.0

    for frame_pair in global_motion_vector:

        if frame_pair is not None and i<len(global_motion_vector)-2:
            for kp_displacement in frame_pair:
                #transformation_array[i][0]=transformation_array[i-1][0]+kp_displacement[0]
                #transformation_array[i][1]=transformation_array[i-1][1]+kp_displacement[1]
                transformation_array[i][0]+=kp_displacement[0]
                transformation_array[i][1]+=kp_displacement[1]

            transformation_array[i][0]=(transformation_array[i][0]/len(frame_pair))
            transformation_array[i][1]=(transformation_array[i][1]/len(frame_pair))

        i+=1

    print(transformation_array)

    trajectoire_array=np.empty([len(transformation_array), 2])
    trajectoire_array[0][0]=0.0
    trajectoire_array[0][1]=0.0

    j=0
    for value in transformation_array:
        trajectoire_array[j][0]=trajectoire_array[j-1][0]+transformation_array[j][0]
        trajectoire_array[j][1]=trajectoire_array[j-1][1]+transformation_array[j][1]
        j+=1

    plt.plot(trajectoire_array)
    plt.title('Trajectoire')
    plt.xlabel('Frame number')
    plt.ylabel('Variation in pixels')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.savefig('../public/trajectoire_array.png')
    plt.show()
    plt.close()

    print(trajectoire_array)


    plt.plot(transformation_array)
    plt.title('Transformation')
    plt.xlabel('Frame number')
    plt.ylabel('Variation in pixels')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.savefig('../public/transformation_array.png')
    plt.show()
    plt.close()


    optimal_size=cv2.getOptimalDFTSize(len(trajectoire_array))
    #trajectoire_array=np.resize(trajectoire_array,(optimal_size,2))

    print(trajectoire_array)


    trajectoire_dft_array_x=np.empty(trajectoire_array[:, [0]].shape)
    trajectoire_dft_array_y=np.empty(trajectoire_array[:, [1]].shape)

    cv2.dft(trajectoire_array[:,[0]], trajectoire_dft_array_x, cv2.DFT_SCALE)
    cv2.dft(trajectoire_array[:,[1]], trajectoire_dft_array_y, cv2.DFT_SCALE)

    #trajectoire_dft_array_x=np.fft.fft(trajectoire_array[:,[0]])
    #trajectoire_dft_array_y=np.fft.fft(trajectoire_array[:,[1]])


    print(trajectoire_dft_array_x)

    plt.plot(trajectoire_dft_array_x)
    plt.title('trajectoire_dft_array_x')
    plt.xlabel('Frame number')
    plt.ylabel('Frequency')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.savefig('../public/trajectoire_dft_array_x.png')
    plt.show()
    plt.close()


    print("########################################################################################################################################")
    #for index,value in enumerate(trajectoire_dft_array_x):
    #    if value>100 or value<-100:
    #        trajectoire_dft_array_x[index]=0

    #for index,value in enumerate(trajectoire_dft_array_y):
    #    if value>100 or value<-100:
    #        trajectoire_dft_array_y[index]=0

    trajectoire_dft_array_x=scipy.ndimage.filters.gaussian_filter(trajectoire_dft_array_x,6,0)
    trajectoire_dft_array_y=scipy.ndimage.filters.gaussian_filter(trajectoire_dft_array_y,6,0)

    plt.plot(trajectoire_dft_array_x)
    plt.title('trajectoire_dft_array_x after filter')
    plt.xlabel('Frame number')
    plt.ylabel('Frequency')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.savefig('../public/trajectoire_dft_array_x_after_filter.png')
    plt.show()
    plt.close()

    print(trajectoire_dft_array_x)


    filtered_trajectoire=np.empty(trajectoire_array.shape)
    filtered_trajectoire_x=np.empty(trajectoire_dft_array_x.shape)
    filtered_trajectoire_y=np.empty(trajectoire_dft_array_y.shape)


    cv2.idft(trajectoire_dft_array_x, filtered_trajectoire_x, cv2.DFT_SCALE)
    cv2.idft(trajectoire_dft_array_y, filtered_trajectoire_y, cv2.DFT_SCALE)

    #filtered_trajectoire_x=np.fft.ifft(trajectoire_dft_array_x)
    #filtered_trajectoire_y=np.fft.ifft(trajectoire_dft_array_y)

    #for index, value in enumerate(filtered_trajectoire_x):
    #    if index<len(global_motion_vector):
    #        filtered_trajectoire_x[index]= value

    #for index, value in enumerate(filtered_trajectoire_y):
    #    if index<len(global_motion_vector):
    #        filtered_trajectoire_y[index]= value

    filtered_trajectoire[:,[0]]=filtered_trajectoire_x
    filtered_trajectoire[:,[1]]=filtered_trajectoire_y
    print(filtered_trajectoire.shape)
    print(filtered_trajectoire)


    plt.plot(filtered_trajectoire)
    plt.title('Trajectoire after filtering')
    plt.xlabel('Frame number')
    plt.ylabel('Variation in pixels')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.savefig('../public/filtered_trajectoire.png')
    plt.show()
    plt.close()


    # Here we compute the global correction vector #
    #buffered_value=[]
    for index,value in enumerate(global_correction_vector):
        if index>0 and index<len(global_correction_vector):
            global_correction_vector[index][0]=transformation_array[index][0]+(filtered_trajectoire[index][0]-trajectoire_array[index][0])
            global_correction_vector[index][1]=transformation_array[index][1]+(filtered_trajectoire[index][1]-trajectoire_array[index][1])

        #buffered_value=value

    #j=1
    #for value in filtered_trajectoire:
    #    if j<len(global_motion_vector):
    #        filtered_global_motion_vector[j][0]=filtered_trajectoire[j][0]-filtered_trajectoire[j-1][0]
    #        filtered_global_motion_vector[j][1]=filtered_trajectoire[j][1]-filtered_trajectoire[j-1][1]

    #print(filtered_global_motion_vector)
    #for item in filtered_global_motion_vector:
    #    file_after_filtering.write("%s\n" % item)
    #raise SystemExit(0)
    print(global_correction_vector)
    return global_correction_vector
