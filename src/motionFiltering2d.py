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

    afp_array=np.empty([len(global_motion_vector), 2])
    i=1
    afp_array[0][0]=0.0
    afp_array[0][1]=0.0
    for frame_pair in global_motion_vector:
        for kp_displacement in frame_pair:
            if i<len(global_motion_vector):
                #afp_array[i][0]=afp_array[i-1][0]+kp_displacement[0]
                #afp_array[i][1]=afp_array[i-1][1]+kp_displacement[1]
                afp_array[i][0]+=kp_displacement[0]
                afp_array[i][1]+=kp_displacement[1]

        if i<len(global_motion_vector):
            afp_array[i][0]=(afp_array[i][0]/len(frame_pair))+afp_array[i-1][0]
            afp_array[i][1]=(afp_array[i][1]/len(frame_pair))+afp_array[i-1][1]

        i+=1


    plt.plot(afp_array)
    plt.title('Absolute frame positions')
    plt.xlabel('Frame number')
    plt.ylabel('Variation in pixels')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.savefig('../public/afp_array.png')
    plt.show()
    plt.close()


    optimal_size=cv2.getOptimalDFTSize(len(afp_array))
    afp_array=np.resize(afp_array,(optimal_size,2))

    print(afp_array)

    dft = cv2.dft(afp_array)
    #dft_shift = np.fft.fftshift(dft)
    dft_shift=dft

    dft_shift=scipy.ndimage.filters.gaussian_filter(dft_shift,6,0)
    #magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

    #rows, cols = afp_array.shape
    #crow,ccol = rows/2 , cols/2

    # create a mask first, center square is 1, remaining all zeros
    #mask = np.zeros((rows,cols,2),np.uint8)
    #mask[crow-30:crow+30, ccol-30:ccol+30] = 1

    # apply mask and inverse DFT
    fshift = dft_shift
    f_ishift = np.fft.ifftshift(fshift)
    filtered_afp = cv2.idft(f_ishift)
    #filtered_afp = cv2.magnitude(filtered_afp[:,:,0],filtered_afp[:,:,1])






    #filtered_afp=np.empty(afp_array.shape)



    print(filtered_afp.shape)
    print(filtered_afp)


    plt.plot(filtered_afp)
    plt.title('Absolute frame positions after filtering')
    plt.xlabel('Frame number')
    plt.ylabel('Variation in pixels')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.savefig('../public/filtered_afp.png')
    plt.show()
    plt.close()

    # Here we compute the global correction vector #
    #buffered_value=[]
    for index,value in enumerate(filtered_afp):
        print(value)
        if index>0 and index<len(global_correction_vector):
            global_correction_vector[index][0]=value[0]-afp_array[index][0]
            global_correction_vector[index][1]=value[1]-afp_array[index][1]

        #buffered_value=value

    #j=1
    #for value in filtered_afp:
    #    if j<len(global_motion_vector):
    #        filtered_global_motion_vector[j][0]=filtered_afp[j][0]-filtered_afp[j-1][0]
    #        filtered_global_motion_vector[j][1]=filtered_afp[j][1]-filtered_afp[j-1][1]

    #print(filtered_global_motion_vector)
    #for item in filtered_global_motion_vector:
    #    file_after_filtering.write("%s\n" % item)
    #raise SystemExit(0)
    print(global_correction_vector)
    raise SystemExit(0)
    return global_correction_vector
