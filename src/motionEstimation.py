def motion_correction (cap, cv2):
    global_correction_vector=[]
    frame_array=[]
    i=0
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
                        #draw_circle_keypoints(cv2, img1, img2, kp1, kp2, match, frame)

                    ####### TO DO : FILTRATE THE DISPLACEMENT TO ONLY KEEEP THE LOW FREQUENCY WITH FPS ########
                    #
                    #
                    #
                    ####### TO DO : FILTRATE THE DISPLACEMENT TO ONLY KEEEP THE LOW FREQUENCY WITH FPS ########

                    #motion_vector=motion_filtering(motion_vector)

                    # Here we compute the correction vector based and the mean value of keypoints displacement #

                    correction_vector=compute_correction_vector(motion_vector)

                    # We build a list keeping track of the correction vector needed for every pair of frame #
                    global_correction_vector.append(correction_vector)

            i+=1
        else:
            i+=1
            break
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
    

def motion_filtering(motion_vector):
    ## TO DO ##
    print("prout")

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
