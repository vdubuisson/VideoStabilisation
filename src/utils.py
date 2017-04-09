import cv2

def create_stabilized_path(path, extension):

    path_stabilized=""
    portion_indice=0
    for portion in path.split('.')[0:len(path.split('.'))-1]:
        if not portion:
            portion="."
        if portion_indice== len(path.split('.'))-2:
            portion=portion+"_stabilized"
        path_stabilized+=portion
        portion_indice+=1

    path_stabilized=path_stabilized+extension
    return path_stabilized


def display_two_vids(vid1, vid2, max_number_frames, videoFps):
    j=0
    while ((vid1.isOpened() or vid2.isOpened()) and j<max_number_frames):

        ret_initial, frame_initial=vid1.read()
        ret_stabilized, frame_stabilized=vid2.read()

        if ret_initial:
            cv2.imshow("Initial video:",frame_initial)
        if ret_stabilized:
            cv2.imshow("Stabilized video:",frame_stabilized)
        if cv2.waitKey(int((1/videoFps)*1000)) & 0xFF == ord('q'):
            break
        j+=1
