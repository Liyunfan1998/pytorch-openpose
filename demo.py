import json, os
import pandas as pd
import cv2

from src.body import Body

body_estimation = Body('model/body_pose_model.pth')


# hand_estimation = Hand('model/hand_pose_model.pth')

def process_one_img(img):
    # test_image = 'images/demo.jpg'
    oriImg = cv2.imread(img)  # B,G,R order
    candidate, subset = body_estimation(oriImg)
    # canvas = copy.deepcopy(oriImg)
    # canvas = util.draw_bodypose(canvas, candidate, subset)
    joints_2d = []
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            xyc = [-1, -1, -1] if index == -1 else candidate[index][0:3]
            joints_2d.extend(xyc)

    '''
    # detect hand
    hands_list = util.handDetect(candidate, subset, oriImg)
    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
        # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
        # if is_left:
            # plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
            # plt.show()
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        # else:
        #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
        #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
        #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        #     print(peaks)
        all_hand_peaks.append(peaks)
    
    canvas = util.draw_handpose(canvas, all_hand_peaks)
    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.show()
    '''
    return joints_2d


def process_one_subject_action_idx(subject_action_idx_dir, subject, action, idx):
    pose_keypoints_2d = []
    csvpath = './output-h36m/' + subject + '_' + action + '_' + idx + '.csv'
    # Skip Duplicate
    if os.path.isfile(csvpath):
        print('# Skip Duplicate', csvpath)
        return
        # save the 2d-pose results to JSON as the OpenPose output format
    cnt = 0
    for root, dirs, files in os.walk(subject_action_idx_dir):
        for file in files:
            img_input = root + '/' + file
            # print(img_input)
            seq = file[:-4]
            pose_keypoints_2d.append([seq, process_one_img(img_input)])
            cnt += 1
    pose_keypoints_2d = pd.DataFrame(pose_keypoints_2d)
    # df.to_csv(out_path, index=False, columns=col)
    #
    pose_keypoints_2d.to_csv(csvpath, index=False)
    print("***", csvpath)


def process_one_subject_action(subject_action_dir, subject, action):
    for root, dirs, files in os.walk(subject_action_dir + '/imageSequence'):
        for dir in dirs:
            subject_action_idx_dir = root + '/' + dir
            # print(subject_action_idx_dir)
            process_one_subject_action_idx(subject_action_idx_dir, subject, action, dir)


def process_one_subject(subject_dir, subject):
    for root, dirs, files in os.walk(subject_dir):
        for dir in dirs:
            subject_action_dir = root + '/' + dir
            # print(subject_action_dir)
            process_one_subject_action(subject_action_dir, subject, dir)


if __name__ == "__main__":
    base_dir = '/linuxExtraHD/h36m-fetch/processed/'
    for root, dirs, files in os.walk(base_dir):
        for dir in dirs:
            subject_dir = root + dir
            # print(subject_dir)
            process_one_subject(subject_dir, dir)
            # exit(213)
