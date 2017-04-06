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

        if frame_pair is not None:
            for kp_displacement in frame_pair:
                if i<len(global_motion_vector):
                    #transformation_array[i][0]=transformation_array[i-1][0]+kp_displacement[0]
                    #transformation_array[i][1]=transformation_array[i-1][1]+kp_displacement[1]
                    transformation_array[i][0]+=kp_displacement[0]
                    transformation_array[i][1]+=kp_displacement[1]

            if i<len(global_motion_vector):
                transformation_array[i][0]=(transformation_array[i][0]/len(frame_pair))
                transformation_array[i][1]=(transformation_array[i][1]/len(frame_pair))
        else:
            transformation_array[i]=transformation_array[i-1]

        if i<len(global_motion_vector) and (transformation_array[i][0]>1000 or transformation_array[i][1]>1000):
            print(i)
            print(transformation_array)
            raise SystemExit(0)
        i+=1

    trajectoire_array=np.empty([len(transformation_array), 2])
    trajectoire_array[0][0]=0.0
    trajectoire_array[0][1]=0.0

    j=1
    for value in transformation_array:
        if j<len(trajectoire_array):
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


    optimal_size=cv2.getOptimalDFTSize(len(transformation_array))
    transformation_array=np.resize(transformation_array,(optimal_size,2))

    print(transformation_array)


    transformation_dft_array_x=np.empty(transformation_array[:, [0]].shape)
    transformation_dft_array_y=np.empty(transformation_array[:, [1]].shape)

    cv2.dft(transformation_array[:,[0]], transformation_dft_array_x, cv2.DFT_SCALE)
    cv2.dft(transformation_array[:,[1]], transformation_dft_array_y, cv2.DFT_SCALE)

    #transformation_dft_array_x=np.fft.fft(transformation_array[:,[0]])
    #transformation_dft_array_y=np.fft.fft(transformation_array[:,[1]])


    print(transformation_dft_array_x)

    plt.plot(transformation_dft_array_x)
    plt.title('transformation_dft_array_x')
    plt.xlabel('Frame number')
    plt.ylabel('Frequency')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.savefig('../public/transformation_dft_array_x.png')
    plt.show()
    plt.close()


    print("########################################################################################################################################")
    #for index,value in enumerate(transformation_dft_array_x):
    #    if value>100 or value<-100:
    #        transformation_dft_array_x[index]=0

    #for index,value in enumerate(transformation_dft_array_y):
    #    if value>100 or value<-100:
    #        transformation_dft_array_y[index]=0

    transformation_dft_array_x=scipy.ndimage.filters.gaussian_filter(transformation_dft_array_x,6,0)
    transformation_dft_array_y=scipy.ndimage.filters.gaussian_filter(transformation_dft_array_y,6,0)

    plt.plot(transformation_dft_array_x)
    plt.title('transformation_dft_array_x after filter')
    plt.xlabel('Frame number')
    plt.ylabel('Frequency')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.savefig('../public/transformation_dft_array_x_after_filter.png')
    plt.show()
    plt.close()

    print(transformation_dft_array_x)


    filtered_transformation=np.empty(transformation_array.shape)
    filtered_transformation_x=np.empty(transformation_dft_array_x.shape)
    filtered_transformation_y=np.empty(transformation_dft_array_y.shape)


    cv2.idft(transformation_dft_array_x, filtered_transformation_x, cv2.DFT_SCALE)
    cv2.idft(transformation_dft_array_y, filtered_transformation_y, cv2.DFT_SCALE)

    #filtered_transformation_x=np.fft.ifft(transformation_dft_array_x)
    #filtered_transformation_y=np.fft.ifft(transformation_dft_array_y)

    #for index, value in enumerate(filtered_transformation_x):
    #    if index<len(global_motion_vector):
    #        filtered_transformation_x[index]= value

    #for index, value in enumerate(filtered_transformation_y):
    #    if index<len(global_motion_vector):
    #        filtered_transformation_y[index]= value

    filtered_transformation[:,[0]]=filtered_transformation_x
    filtered_transformation[:,[1]]=filtered_transformation_y
    print(filtered_transformation.shape)
    print(filtered_transformation)


    plt.plot(filtered_transformation)
    plt.title('Transformation after filtering')
    plt.xlabel('Frame number')
    plt.ylabel('Variation in pixels')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.savefig('../public/filtered_transformation.png')
    plt.show()
    plt.close()



    # Here we compute the global correction vector #
    #buffered_value=[]
    for index,value in enumerate(filtered_transformation):
        if index>0 and index<len(global_correction_vector):
            global_correction_vector[index][0]=value[0]-transformation_array[index][0]
            global_correction_vector[index][1]=value[1]-transformation_array[index][1]

        #buffered_value=value

    #j=1
    #for value in filtered_transformation:
    #    if j<len(global_motion_vector):
    #        filtered_global_motion_vector[j][0]=filtered_transformation[j][0]-filtered_transformation[j-1][0]
    #        filtered_global_motion_vector[j][1]=filtered_transformation[j][1]-filtered_transformation[j-1][1]

    #print(filtered_global_motion_vector)
    #for item in filtered_global_motion_vector:
    #    file_after_filtering.write("%s\n" % item)
    #raise SystemExit(0)
    print(global_correction_vector)
    return global_correction_vector
