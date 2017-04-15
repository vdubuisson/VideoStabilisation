def image_warping(cap2, fourcc, out, buffer_frame, max_number_frames, global_correction_vector, np, cv2, frameWidth, frameHeight, border_type ):
    counter=0
    while(cap2.isOpened() and counter<max_number_frames):
        ret, frame = cap2.read()

        if ret==True and counter<len(global_correction_vector):

            print("Counter:")
            print(counter)
            print("global_correction_vector at this frame:")

            print(global_correction_vector[counter])

            shiftX = int(global_correction_vector[counter][0])
            shiftY = int(global_correction_vector[counter][1])

            # We create a numpy matrix to shift the frame according to the correction vector #
            shiftMatrix = np.float32([[1, 0, shiftX], [0, 1, shiftY]])

            # We shift the frame with the matrix #
            newFrame = cv2.warpAffine(frame, shiftMatrix, (frameWidth, frameHeight))

            # Borders a managed by using previous frame pixels #
            if border_type=="replace":
                if shiftX > 0 :
                    # Left border #
                    newFrame[0:frameHeight, 0:shiftX] = buffer_frame[0:frameHeight, 0:shiftX].copy()
                elif shiftX < 0 :
                    # Right border #
                    newFrame[0:frameHeight, frameWidth+shiftX:frameWidth] = buffer_frame[0:frameHeight, frameWidth+shiftX:frameWidth].copy()
                if shiftY > 0 :
                    # Top border #
                    newFrame[0:shiftY, 0:frameWidth] = buffer_frame[0:shiftY, 0:frameWidth].copy()
                elif shiftY < 0 :
                    # Bottom border #
                    newFrame[frameHeight+shiftY:frameHeight, 0:frameWidth] = buffer_frame[frameHeight+shiftY:frameHeight, 0:frameWidth].copy()

                # We apply a blur on the new borders
                blurredFrame = newFrame.copy()
                # kernelX = np.float32([[1,1,1,1,1]])/5
                # kernelY = np.float32([[0],[0.1],[0.25],[0.75],[1],[0.75],[0.25],[0.1],[0]])/3.2
                if shiftX > 0 :
                    # Left border #
                    blurredFrame[0:frameHeight, 0:shiftX+2] = cv2.GaussianBlur(blurredFrame[0:frameHeight, 0:shiftX+2], (5,5), 0, 0, cv2.BORDER_REPLICATE)
                elif shiftX < 0 :
                    # Right border #
                    blurredFrame[0:frameHeight, frameWidth+shiftX-2:frameWidth] = cv2.GaussianBlur(blurredFrame[0:frameHeight, frameWidth+shiftX-2:frameWidth], (5,5), 0, 0, cv2.BORDER_REPLICATE)
                if shiftY > 0 :
                    # Top border #
                    blurredFrame[0:shiftY+2, 0:frameWidth] = cv2.GaussianBlur(blurredFrame[0:shiftY+2, 0:frameWidth], (5,5), 0, 0, cv2.BORDER_REPLICATE)
                elif shiftY < 0 :
                    # Bottom border #
                    blurredFrame[frameHeight+shiftY-2:frameHeight, 0:frameWidth] = cv2.GaussianBlur(blurredFrame[frameHeight+shiftY-2:frameHeight, 0:frameWidth], (5,5), 0, 0, cv2.BORDER_REPLICATE)

            # Border are filled with white pixels #
            elif border_type=="white":
                if shiftX > 0 :
                    # Left border #
                    newFrame[0:frameHeight, 0:shiftX] = np.full((frameHeight, frameWidth, 3), 255, dtype=int)[0:frameHeight, 0:shiftX]
                    #newFrame[0:frameHeight, 0:shiftX+2] = cv2.blur(newFrame[0:frameHeight, 0:shiftX+2], (5,5))
                elif shiftX < 0 :
                    # Right border #
                    newFrame[0:frameHeight, frameWidth+shiftX:frameWidth] = np.full((frameHeight, frameWidth, 3), 255, dtype=int)[0:frameHeight, frameWidth+shiftX:frameWidth]
                    #newFrame[0:frameHeight, frameWidth+shiftX-2:frameWidth] = cv2.blur(newFrame[0:frameHeight, frameWidth+shiftX-2:frameWidth], (5,5))
                if shiftY > 0 :
                    # Top border #
                    newFrame[0:shiftY, 0:frameWidth] = np.full((frameHeight, frameWidth, 3), 255, dtype=int)[0:shiftY, 0:frameWidth]
                    #newFrame[0:shiftY+2, 0:frameWidth] = cv2.blur(newFrame[0:shiftY+2, 0:frameWidth], (5,5))
                elif shiftY < 0 :
                    # Bottom border #
                    newFrame[frameHeight+shiftY:frameHeight, 0:frameWidth] = np.full((frameHeight, frameWidth, 3), 255, dtype=int)[frameHeight+shiftY:frameHeight, 0:frameWidth]
                    #newFrame[frameHeight+shiftY-2:frameHeight, 0:frameWidth] = cv2.blur(newFrame[frameHeight+shiftY-2:frameHeight, 0:frameWidth], (5,5))

            # Border are filled with white pixels #
            elif border_type=="black":
                if shiftX > 0 :
                    # Left border #
                    newFrame[0:frameHeight, 0:shiftX] = np.full((frameHeight, frameWidth, 3), 0, dtype=int)[0:frameHeight, 0:shiftX]
                    #newFrame[0:frameHeight, 0:shiftX+2] = cv2.blur(newFrame[0:frameHeight, 0:shiftX+2], (5,5))
                elif shiftX < 0 :
                    # Right border #
                    newFrame[0:frameHeight, frameWidth+shiftX:frameWidth] = np.full((frameHeight, frameWidth, 3), 0, dtype=int)[0:frameHeight, frameWidth+shiftX:frameWidth]
                    #newFrame[0:frameHeight, frameWidth+shiftX-2:frameWidth] = cv2.blur(newFrame[0:frameHeight, frameWidth+shiftX-2:frameWidth], (5,5))
                if shiftY > 0 :
                    # Top border #
                    newFrame[0:shiftY, 0:frameWidth] = np.full((frameHeight, frameWidth, 3), 0, dtype=int)[0:shiftY, 0:frameWidth]
                    #newFrame[0:shiftY+2, 0:frameWidth] = cv2.blur(newFrame[0:shiftY+2, 0:frameWidth], (5,5))
                elif shiftY < 0 :
                    # Bottom border #
                    newFrame[frameHeight+shiftY:frameHeight, 0:frameWidth] = np.full((frameHeight, frameWidth, 3), 0, dtype=int)[frameHeight+shiftY:frameHeight, 0:frameWidth]
                    #newFrame[frameHeight+shiftY-2:frameHeight, 0:frameWidth] = cv2.blur(newFrame[frameHeight+shiftY-2:frameHeight, 0:frameWidth], (5,5))

            # if border_type=="test":
            #     if shiftY < 0 :
            #         # Bottom border
            #         im1_gray = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY)
            #         im2 = buffer_frame.copy()
            #         im2[0:frameHeight+shiftY, 0:frameWidth] = np.full((frameHeight, frameWidth, 3), 0, dtype=int)[0:frameHeight+shiftY, 0:frameWidth]
            #         im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
            #         warp_matrix = np.eye(2, 3, dtype=np.float32)
            #         num_iter = 50
            #         term_eps = 1e-10
            #         criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iter,  term_eps)
            #         (cc, warp_matrix) = cv2.findTransformECC(im1_gray,im2_gray,warp_matrix, cv2.MOTION_AFFINE, criteria)
            #         im2_aligned = cv2.warpAffine(im2, warp_matrix, (frameWidth,frameHeight),flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
            #         newFrame[frameHeight+shiftY:frameHeight, 0:frameWidth] = im2_aligned[frameHeight+shiftY:frameHeight, 0:frameWidth].copy()

            # We save the corrected frame #
            if border_type=='replace':
                out.write(blurredFrame)
                buffer_frame=blurredFrame.copy()
            else:
                out.write(newFrame)
                buffer_frame=newFrame
        else:
            # We release the second capture #
            cap2.release()
            out.release()

        counter+=1
