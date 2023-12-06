import os.path
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import MediaPipeProcess.keypoint_extract as md
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import MediaPipeProcess.create_point as my_data

# setup plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
mp_hands = mp.solutions.hands.Hands(static_image_mode=False,
                                    max_num_hands=2,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)
mp_pose = mp.solutions.pose.Pose(model_complexity=2, enable_segmentation=True,
                                 min_detection_confidence=0.5)


def get_list_frame(path_video):
    result = []
    cap = cv2.VideoCapture(path_video)
    while cap.isOpened():
        # read webcam image
        success, image = cap.read()

        # skip empty frames
        if not success:
            break
        left_hand_landmarks, right_hand_landmarks = md.extract_hand_landmark(image, mp_hands)
        body_point = md.extract_body_landmark(image, mp_pose)
        result.append([left_hand_landmarks, right_hand_landmarks, body_point])
    cap.release()
    return result


def concate_array(left_hand_landmarks, right_hand_landmarks, body_point):
    a1 = np.array(left_hand_landmarks).reshape(-1)
    a2 = np.array(right_hand_landmarks).reshape(-1)
    a3 = np.array(body_point).reshape(-1)
    result = np.concatenate((a1, a2, a3), axis=None)
    return result


def check_zeros(list_data):
    data = np.array(list_data)
    if np.all(data == 0):
        return True
    return False


def view_data(list_data):
    for data in list_data:
        md.plot_world_landmarks(ax, data[0], data[1], data[2])


def write_data(path_dir_to_save, path_source_file, name_file_save):
    try:
        list_fr = get_list_frame(path_source_file)
        X = []
        list_index = []
        for i in range(len(list_fr)):
            if not check_zeros(list_fr[i][0]) and not check_zeros(list_fr[i][1]):
                X.append(concate_array(list_fr[i][0], list_fr[i][1], list_fr[i][2]))
                list_index.append(i)
        X_new = np.array(X)
        K = 20
        kmeans = KMeans(n_clusters=K)
        kmeans.fit(X_new)
        # Lấy ra trung tâm của từng nhóm
        cluster_centers = kmeans.cluster_centers_
        # Tính khoảng cách giữa mỗi vector và trung tâm của nhóm của nó
        distances = cdist(X_new, cluster_centers, 'euclidean')
        # print(distances[0])
        nearest_indices = np.argmin(distances, axis=0)
        index = np.sort(nearest_indices)
        data = []
        for i in index:
            data.append(list_fr[list_index[i]])
        data_save = np.asarray(data, dtype="object")
        np.save(os.path.join(path_dir_to_save, name_file_save), data_save)
        print("ok write from file  : " + path_source_file)
    except:
        print("error write : " + path_source_file)


def general_data(path_source, path_to_save, number_sample):
    from LSTM.read_data import create_directory
    list_dir = os.listdir(path_source)
    for sub_dir in list_dir:
        try:
            if not os.path.exists(os.path.join(path_to_save, sub_dir)):
                create_directory(path_to_save, sub_dir)
            path_file = os.path.join(path_source, sub_dir) + "/0.npy"
            if os.path.isfile(path_file):
                source_sample = np.load(path_file, allow_pickle=True)
                #frame_0 = md.extract_landmarks_by_img("/image_signer/test.png")
                frame_0 = my_data.create_frame_0(source_sample)
                for index in range(30, number_sample + 30):
                    new_file = os.path.join(path_to_save, sub_dir) + "/" + str(index) + ".npy"
                    result = []
                    # print("index: ", index)
                    result.append(frame_0)
                    for i in range(1, 20):
                        frame_i = my_data.create_frame_t(i, source_sample, frame_0)
                        result.append(frame_i)
                    data_save = np.asarray(result, dtype="object")
                    np.save(new_file, data_save)
        except:
            print("error when general data_test")


def get_files_in_directory(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_list.append(file_path)
    return file_list


def general_data_v1(path_source, path_to_save, number_sample):
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    from LSTM.read_data import create_directory
    list_dir = os.listdir(path_source)
    for sub_dir in list_dir:
        try:
            if not os.path.exists(os.path.join(path_to_save, sub_dir)):
                create_directory(path_to_save, sub_dir)
                print(sub_dir)
            path_file = os.path.join(path_source, sub_dir) + "/0.npy"
            if os.path.isfile(path_file):
                source_sample = np.load(path_file, allow_pickle=True)
                list_img = get_files_in_directory('/home/khanhlinux/ML/autodata_sign/image_signer')
                num_name = 0
                for img in list_img:
                    if num_name < number_sample:
                        for index in range(10):
                            frame_0 = md.extract_landmarks_by_img(img, hands, pose)
                            new_file = os.path.join(path_to_save, sub_dir) + "/" + str(num_name) + ".npy"
                            result = []
                            # print("index: ", index)
                            result.append(source_sample[0])
                            for i in range(1, 20):
                                frame_i = my_data.create_frame_t(i, source_sample, frame_0)
                                result.append(frame_i)
                            data_save = np.asarray(result, dtype="object")
                            np.save(new_file, data_save)
                            num_name += 1
                if (num_name < number_sample):
                    frame_0 = my_data.create_frame_0(source_sample)
                    for index in range(num_name, number_sample):
                        new_file = os.path.join(path_to_save, sub_dir) + "/" + str(index) + ".npy"
                        result = []
                        # print("index: ", index)
                        result.append(frame_0)
                        for i in range(1, 20):
                            frame_i = my_data.create_frame_t(i, source_sample, frame_0)
                            result.append(frame_i)
                        data_save = np.asarray(result, dtype="object")
                        np.save(new_file, data_save)
            print("done :" +str(num_name))
        except:
            print("error when general data_test")


def general_data_v2(path_source, path_to_save, number_sample):
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    from LSTM.read_data import create_directory
    list_dir = os.listdir(path_source)
    for sub_dir in list_dir:
        try:
            if not os.path.exists(os.path.join(path_to_save, sub_dir)):
                create_directory(path_to_save, sub_dir)
                print (sub_dir)
            for file_name in os.listdir(os.path.join(path_source, sub_dir)):
                path_file = os.path.join(os.path.join(path_source, sub_dir), file_name)
                if os.path.isfile(path_file):
                    source_sample = np.load(path_file, allow_pickle=True)
                    list_img = get_files_in_directory('/home/khanhlinux/ML/autodata_sign/image_signer')
                    num_name = 0
                    for img in list_img:
                        if num_name < number_sample:
                            frame_0 = md.extract_landmarks_by_img(img, hands, pose)
                            for index in range(5):
                                new_file = os.path.join(path_to_save, sub_dir) + "/" + str(num_name) + ".npy"
                                result = []
                                # print("index: ", index)
                                result.append(frame_0)
                                for i in range(1, 20):
                                    frame_i = my_data.create_frame_t(i, source_sample, frame_0)
                                    result.append(frame_i)
                                data_save = np.asarray(result, dtype="object")
                                np.save(new_file, data_save)
                                num_name += 1
                    if (num_name < number_sample):
                        frame_0 = my_data.create_frame_0(source_sample)
                        for index in range(num_name, number_sample):
                            new_file = os.path.join(path_to_save, sub_dir) + "/" + str(index) + ".npy"
                            result = []
                            # print("index: ", index)
                            result.append(frame_0)
                            for i in range(1, 20):
                                frame_i = my_data.create_frame_t(i, source_sample, frame_0)
                                result.append(frame_i)
                            data_save = np.asarray(result, dtype="object")
                            np.save(new_file, data_save)
        except:
            print("error when general data_test")

def general_data_v3(path_source, path_to_save, number_sample):
    '''mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)'''
    from LSTM.read_data import create_directory
    list_dir = os.listdir(path_source)
    for sub_dir in list_dir:
        try:
            if not os.path.exists(os.path.join(path_to_save, sub_dir)):
                create_directory(path_to_save, sub_dir)
                print (sub_dir)
            #for file_name in os.listdir(os.path.join(path_source, sub_dir)):
            path_file = os.path.join(os.path.join(path_source, sub_dir), '0.npy')
            print(path_file)
            if os.path.isfile(path_file):
                source_sample = np.load(path_file, allow_pickle=True)
                list_img = get_files_in_directory('/home/khanhlinux/ML/autodata_sign/image_signer')
                num_name = 0
                '''for img in list_img:
                    if num_name < number_sample:
                        frame_0 = md.extract_landmarks_by_img(img, hands, pose)
                        for index in range(5):
                            new_file = os.path.join(path_to_save, sub_dir) + "/" + str(num_name) + ".npy"
                            result = []
                            # print("index: ", index)
                            result.append(frame_0)
                            for i in range(1, 20):
                                frame_i = my_data.create_frame_t(i, source_sample, frame_0)
                                result.append(frame_i)
                            data_save = np.asarray(result, dtype="object")
                            np.save(new_file, data_save)
                            num_name += 1'''
                if (num_name < number_sample):
                    frame_0 = my_data.create_frame_0(source_sample)
                    for index in range(num_name, number_sample):
                        new_file = os.path.join(path_to_save, sub_dir) + "/" + str(index) + ".npy"
                        result = []
                        # print("index: ", index)
                        result.append(frame_0)
                        for i in range(1, 20):
                            frame_i = my_data.create_frame_t(i, source_sample, frame_0)
                            result.append(frame_i)
                        data_save = np.asarray(result, dtype="object")
                        np.save(new_file, data_save)
        except:
            print("error when general data_test")
