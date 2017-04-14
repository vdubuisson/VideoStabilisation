import numpy as np
def motion_estimation(frame_array,cv2,j, frameWidth, frameHeight):
    # motion_vector containes the displacement of all the keypoints for the current pair of frame #
    motion_vector=[]
    img1=frame_array[j-1]
    img2=frame_array[j]

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

        #for index, match in enumerate(matches):
        #    if i<10:

        #        motion_vector.append((kp2[match.trainIdx].pt[0]-kp1[match.queryIdx].pt[0], kp2[match.trainIdx].pt[1]- kp1[match.queryIdx].pt[1]))
        #        print("PROUTTTTTTT")
        #        print((frameWidth/2-kp2[match.trainIdx].pt[0])/frameWidth)
        #        print((frameWidth/2-kp1[match.queryIdx].pt[0])/frameWidth)
        #        print((frameHeight/2-kp2[match.trainIdx].pt[1])/frameHeight)
        #        print((frameHeight/2-kp1[match.queryIdx].pt[1])/frameHeight)
                #draw_circle_keypoints(cv2, img1, img2, kp1, kp2, match)
        #    i+=1

        counter_kp=0
        i=0
        distance_to_mid=0.05
        kp_already_used=np.zeros(len(matches))

        while counter_kp<10 and counter_kp<len(matches):
            #print("frame number")
            #print(j)
            #print("i=")
            #print(i)
            #print("kp_already_used")
            #print(kp_already_used)
            #print("distance_to_mid")
            #print(distance_to_mid)
            #print("counter_kp")
            #print(counter_kp)
            if i<len(matches):
                if (frameWidth/2-kp2[matches[i].trainIdx].pt[0])/frameWidth<distance_to_mid:
                    if(frameWidth/2-kp1[matches[i].queryIdx].pt[0])/frameWidth<distance_to_mid:
                        if (frameHeight/2-kp2[matches[i].trainIdx].pt[1])/frameHeight<distance_to_mid:
                            if(frameHeight/2-kp1[matches[i].queryIdx].pt[1])/frameHeight<distance_to_mid:
                                if kp_already_used[i]==0:
                                    motion_vector.append((kp2[matches[i].trainIdx].pt[0]-kp1[matches[i].queryIdx].pt[0], kp2[matches[i].trainIdx].pt[1]- kp1[matches[i].queryIdx].pt[1]))
                                    kp_already_used[i]=1
                                    counter_kp+=1
                                    i+=1
                                else:
                                    i+=1
                            else:
                                i+=1
                        else:
                            i+=1
                    else:
                        i+=1
                else:
                    i+=1
            else:
                i=0
                distance_to_mid=distance_to_mid*2

        return motion_vector


def draw_circle_keypoints(cv2, img1, img2, kp1, kp2, match):
    # This piece of code is if you want to vizualise the keypoints (useful to debug) #

    img_circle1=img1
    cv2.circle(img_circle1, (int(kp1[match.queryIdx].pt[0]),int(kp1[match.queryIdx].pt[1])), 3, (0,0,255), 1, 8 )

    img_circle2=img2
    cv2.circle(img_circle2, (int(kp2[match.trainIdx].pt[0]),int(kp2[match.trainIdx].pt[1])), 3, (0,0,255), 1, 8 )

    cv2.imshow("img_circle1",img_circle1)
    cv2.imshow("img_circle2",img_circle2)
