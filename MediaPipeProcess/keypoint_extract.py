import cv2
import matplotlib.pyplot as plt
import mediapipe as mp


# Khởi tạo đối tượng Hands
def extract_hand_landmark(img, model):
    left_hand_landmarks = []
    right_hand_landmarks = []
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Nhận dạng các bàn tay
    results = model.process(image_rgb)
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if idx < 2:
                hand_label = results.multi_handedness[idx].classification[0].label
                # Trích xuất tọa độ của các điểm trên bàn tay
                for id, landmark in enumerate(hand_landmarks.landmark):
                    #x = round(landmark.x * img.shape[1], 3)
                    #y = round(landmark.y * img.shape[0], 3)
                    #z = round(landmark.z * img.shape[1], 3)
                    x = round(landmark.x , 3)
                    y = round(landmark.y , 3)
                    z = round(landmark.z, 3)
                    if hand_label == 'Left':
                        left_hand_landmarks.append([x, y, 0])
                    elif hand_label == 'Right':
                        right_hand_landmarks.append([x, y, 0])
                # print(f'{idx}' + "|" + str(len(left_hand_landmarks)) + "  |" + str(len(right_hand_landmarks)))
    if len(left_hand_landmarks) == 0:
        left_hand_landmarks = [[0, 0, 0]] * 21
    if len(right_hand_landmarks) == 0:
        right_hand_landmarks = [[0, 0, 0]] * 21

    return left_hand_landmarks, right_hand_landmarks


def extract_body_landmark(img, model):
    list_pose = []
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = model.process(image_rgb)
    if result.pose_landmarks:
        for idx, landmark in enumerate(result.pose_landmarks.landmark):
            #x = round(landmark.x * img.shape[1], 3)
            #y = round(landmark.y * img.shape[0], 3)
            #z = round(landmark.z * img.shape[1], 3)
            x = round(landmark.x, 3)
            y = round(landmark.y, 3)
            z = round(landmark.z, 3)
            list_pose.append([x, y, 0])
    if len(list_pose) == 0:
        list_pose = [[0, 0, 0]] * 33
    for i in range(17, 33):
        if i != 23 and i != 24:
            list_pose[i] = [0, 0, 0]
    return list_pose


def extract_landmarks_by_img(path_img, hands, pose):
    img = cv2.imread(path_img)
    left_hand_landmarks, right_hand_landmarks = extract_hand_landmark(img, hands)
    list_pose = extract_body_landmark(img, pose)
    return [left_hand_landmarks, right_hand_landmarks, list_pose]


def plot_world_landmarks(ax, left_hand_marks, right_hand_marks, body_land_marks):
    """_summary_
    Args:
        ax: plot axes
        landmarks  mediapipe
        :param body_land_marks:
        :param right_hand_marks:
        :param ax:
        :param left_hand_marks:
    """
    ax.cla()
    # had to flip the z axis
    # ax.set_xlim3d(-1, wight)
    # ax.set_ylim3d(-1, high)
    # ax.set_zlim3d(1,wight)
    hand_segment = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [5, 9, 13, 17], [9, 10, 11, 12], [13, 14, 15, 16],
                    [0, 17, 18, 19, 20]]
    pose_segment = [[14, 12, 11, 13], [12, 24, 23, 11], [0, 1, 2, 3], [0, 4, 5, 6], [10, 9]]
    try:
        for line in hand_segment:
            plotX_l = [left_hand_marks[i][0] for i in line]
            plotY_l = [left_hand_marks[i][1] for i in line]
            plotZ_l = [left_hand_marks[i][2] for i in line]
            plotX_r = [right_hand_marks[i][0] for i in line]
            plotY_r = [right_hand_marks[i][1] for i in line]
            plotZ_r = [right_hand_marks[i][2] for i in line]
            ax.plot(plotX_l, plotY_l, plotZ_l)
            ax.plot(plotX_r, plotY_r, plotZ_r)
        for line in pose_segment:
            plotX = [body_land_marks[i][0] for i in line]
            plotY = [body_land_marks[i][1] for i in line]
            plotZ = [body_land_marks[i][2] for i in line]
            ax.plot(plotX, plotY, plotZ)
        if left_hand_marks[0][0] != 0 or left_hand_marks[0][1] != 0 or left_hand_marks[0][2]:
            X_gate_left = [left_hand_marks[0][0], body_land_marks[14][0]]
            Y_gate_left = [left_hand_marks[0][1], body_land_marks[14][1]]
            Z_gate_left = [left_hand_marks[0][2], body_land_marks[14][2]]
            ax.plot(X_gate_left, Y_gate_left, Z_gate_left)
        if right_hand_marks[0][0] != 0 or right_hand_marks[0][1] != 0 or right_hand_marks[0][2]:
            X_gate_right = [right_hand_marks[0][0], body_land_marks[13][0]]
            Y_gate_right = [right_hand_marks[0][1], body_land_marks[13][1]]
            Z_gate_right = [right_hand_marks[0][2], body_land_marks[13][2]]
            ax.plot(X_gate_right, Y_gate_right, Z_gate_right)

        ax.view_init(elev=90, azim=90, roll=0)
        plt.pause(0.1)
    except:
        print("An exception occurred")
    return
