import os
import cv2


def splitVideo(source_path, save_path, v):
    cap = cv2.VideoCapture(source_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # frame_time = cap.get(cv2.CAP_PROP_POS_MSEC)

        if ret:
            # if count == 20:
            frame_name = f"{v}_{count}.jpg"
            # print(frame_name)
            print(os.path.join(save_path, frame_name))
            cv2.imwrite(os.path.join(save_path, frame_name), frame)
            count += 1

        else:
            break
    cap.release()


if __name__ == '__main__':
    source_path = r"F:\Datasets\script"
    save_path = r"F:\Datasets\script\tmp"
    video_name = r"round"
    video_list = os.listdir(video_name)
    for v in video_list:
        # if v == "output_9__19.mp4":
        source_path = os.path.join(video_name, v)
        # print(source_path)
        splitVideo(source_path, save_path, v[:-4])
    # splitVideo(video_name, save_path, video_name[:-4])
