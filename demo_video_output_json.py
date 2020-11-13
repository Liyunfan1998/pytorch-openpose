import json

import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import os, sys
from src import model
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('model/body_pose_model.pth')
# hand_estimation = Hand('model/hand_pose_model.pth')

print(f"Torch device: {torch.cuda.get_device_name()}")


def process_one(video_input):
    seq = video_input.split('/')[-1].split('.')[
        0]
    output_fname = './output-vid/' + seq + '_openpose_inference_kp.avi'
    print(output_fname)
    if os.path.isfile(output_fname):
        print("file alreay exist, return and process next")
        return
    cap = cv2.VideoCapture(video_input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = cap.get(cv2.CAP_PROP_FPS)
    # cap.set(3, 640)
    # cap.set(4, 480)
    # output_file = cv2.VideoWriter(
    #     filename=output_fname,
    #     # some installation of opencv may not support x264 (due to its license),
    #     # you can try other format (e.g. MPEG)
    #     # apiPreference=1,
    #     fourcc=cv2.VideoWriter_fourcc(*'XVID'),
    #     # fourcc=cv2.VideoWriter_fourcc(*"x264"),
    #     fps=float(frames_per_second),
    #     frameSize=(width, height),
    #     isColor=True)
    people = {}
    people['pose_keypoints_2d'] = {}
    cnt = 0
    while True:
        ret, oriImg = cap.read()
        cnt += 1
        print("frame", cnt)
        if oriImg is None:
            break
        # if cnt % 10 != 0: continue
        candidate, subset = body_estimation(oriImg)
        # save the 2d-pose results to JSON as the OpenPose output format
        people['pose_keypoints_2d'][cnt] = []

        for i in range(18):
            for n in range(len(subset)):
                index = int(subset[n][i])
                xyc = [-1, -1, -1] if index == -1 else candidate[index][0:3]
                people['pose_keypoints_2d'][cnt].extend(xyc)
        print(people['pose_keypoints_2d'][cnt])
        #
        # canvas = copy.deepcopy(oriImg)
        # canvas = util.draw_bodypose(canvas, candidate, subset)
        # canvas = util.draw_bodypose(oriImg, candidate, subset)
        # detect hand
        # hands_list = util.handDetect(candidate, subset, oriImg)
        #
        # all_hand_peaks = []
        # for x, y, w, is_left in hands_list:
        #     peaks = hand_estimation(oriImg[y:y + w, x:x + w, :])
        #     peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
        #     peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
        #     all_hand_peaks.append(peaks)
        #
        # canvas = util.draw_handpose(canvas, all_hand_peaks)
        #
        # cv2.imshow('demo', canvas)  # 一个窗口用以显示原视频
        # output_file.write(canvas)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    cap.release()
    # output_file.release()
    cv2.destroyAllWindows()
    with open("./output-vid/" + seq + ".json", "w") as dump_f:
        json.dump(people, dump_f)


if __name__ == "__main__":
    video_input_dir = './input-vid/'
    for root, dirs, files in os.walk(video_input_dir):
        for file in files:
            video_input = root + file
            print(video_input)
            process_one(video_input)
            # exit(213)
