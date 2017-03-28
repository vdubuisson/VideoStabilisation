def motion_estimation(frame_array,cv2,i):
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
        return motion_vector
