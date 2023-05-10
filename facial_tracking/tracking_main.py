from facenet_pytorch import MTCNN
import torch
import torchvision
import numpy as np
import cv2
from PIL import Image, ImageDraw
from IPython import display
import matplotlib
import os
import sys


def get_frames(video_path):
    video = cv2.VideoCapture(video_path)
    success, snip = video.read()
    framez = []
    while success:
      framez.append(snip)     # save frame as JPEG file
      success,snip = video.read()

    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in framez]
    h = np.array(framez[0][:,:,0]).shape[0]
    w = np.array(framez[0][:,:,0]).shape[1]
    return framez, frames, h, w


def dist_cal(y_cent, x_cent, y, x):
    dist = ((y_cent - y)**2 + (x_cent - x)**2)**0.5
    return dist


def resize(frames_, trim_dim, final_dim=160):
    frames_resized = []
    cent_crop = torchvision.transforms.CenterCrop((trim_dim,trim_dim))
    for i in range(len(frames_cropped)):
        tens_test = Image.fromarray(cv2.cvtColor(frames_cropped[i], cv2.COLOR_BGR2RGB))
        tens_test = cent_crop(tens_test)
        frames_resized.append(torchvision.transforms.functional.resize(tens_test, (final_dim,final_dim)))
    return frames_resized

def get_cropped_frames(frames, h, w, min_face_h=100, min_face_w=100):
    frames_cropped = []
    face_h_avg = 0
    face_w_avg = 0
    for i, frame in enumerate(frames):
        print('\rTracking frame: {}'.format(i + 1), end='')
        # Detect faces
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            box_dists = []
            for box in boxes:
                rect_cord = box.tolist()
                dotx = (rect_cord[2] + rect_cord[0]) / 2
                doty = (rect_cord[1] + rect_cord[3]) / 2
                dist = dist_cal(h/2, w/2, doty, dotx)
                box_dists.append(dist)

            close_box = boxes[np.argmin(box_dists)]
            close_box_cord = close_box.tolist()

            if ((close_box_cord[2] - close_box_cord[0]) > min_face_w) or ((close_box_cord[1] + close_box_cord[3]) > min_face_h):
              dotx = (close_box_cord[2] + close_box_cord[0]) / 2
              doty = (close_box_cord[1] + close_box_cord[3]) / 2
              xy = [(dotx,doty)]

              cropped_face = frames[i][max(int(close_box_cord[1]) - 40, 0): min(int(close_box_cord[3]) + 40, h - 1), max(int(close_box_cord[0]) - 40, 0): min(int(close_box_cord[2]) + 40, w - 1),:]
              face_h_avg += cropped_face.shape[0]
              face_w_avg += cropped_face.shape[1]
              frames_cropped.append(np.array(cropped_face))

    face_w_avg = int(face_w_avg / len(frames))
    face_h_avg = int(face_h_avg / len(frames))

    return frames_cropped, face_w_avg, face_h_avg


if __name__ == '__main__':
    test_deblurred = int(sys.argv[1]) == 0 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(keep_all=True, device=device)
    input_path = '/home/anaveen/Documents/harvard_ws/spring2023/mit6.8301/RobustLipReadingCV/data/'
    if test_deblurred:
        input_path += 'deblurred/'
    else:
        input_path += 'blurred/'
    print(os.listdir(input_path))

    demo_list = ''
    for video_file in os.listdir(input_path):
        framez, frames, h, w = get_frames(input_path + video_file)
        frames_cropped, face_h_avg, face_w_avg = get_cropped_frames(framez, h, w)
        trim_dim = min(face_h_avg,face_w_avg)
        face_frames = resize(frames_cropped, trim_dim)
        dim = face_frames[0].size
        print(dim)

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_tracked = cv2.VideoWriter('/home/anaveen/Documents/harvard_ws/spring2023/mit6.8301/RobustLipReadingCV/deep_lip_reading/media/example/' + video_file, fourcc, 25.0, dim)
        for frame in face_frames:
            video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        video_tracked.release()
        demo_list += video_file + ', ' + " ".join(video_file[:-4].split('_')) + "\n"

    f = open("/home/anaveen/Documents/harvard_ws/spring2023/mit6.8301/RobustLipReadingCV/deep_lip_reading/media/example/demo_list.txt", "w")
    f.write(demo_list[:-1])
    f.close()





print('\nDone')
