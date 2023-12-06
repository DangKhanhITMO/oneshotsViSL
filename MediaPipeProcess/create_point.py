import random
import numpy as np
EPSILON_BODY = 8
EPSILON_HAND_FINGER = 6
EPSILON_EYE = 3
K_BODY = random.uniform(0.8, 1.2)

K_FINGER = K_BODY*0.9
def distance(M1, M2):
    x1, y1, z1 = tuple(M1)
    x2, y2, z2 = tuple(M2)
    d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return d

def create_noise_point(point, epsilon):
    x, y, z = tuple(point)
    ex = random.uniform(-epsilon, epsilon)
    ey = random.uniform(-epsilon, epsilon)
    ez = random.uniform(-0.5, 0.5)
    #return [round(x + ex,3), round(y + ey,3), round(z + ez,3)]
    return [round(x + ex, 3), round(y + ey, 3), 0]


def create_point_by_k(A, B, M, k):
    xA, yA, zA = tuple(A)
    xB, yB, zB = tuple(B)
    xM, yM, zM = tuple(M)
    x = xB - xA
    y = yB - yA
    z = zB - zA
    #return [round(k * x + xM, 3),round( k * y + yM, 3), round(k * z + zM, 3)]
    return [round(k * x + xM, 3), round(k * y + yM, 3), 0]


def calculate_k(A1, B1, A2, B2, M1, N1):
    d1 = distance(A1, B1)
    if d1==0:
        d1=1
        print(B1)
    h1 = distance(M1, N1)
    k1 = h1 / d1
    return k1


def calculate_start_point(A1, B1, A2, B2, M1, N1):
    k0 = K_BODY
    M2 = create_point_by_k(A1, A2, M1, k0)
    k1 = calculate_k(A1, B1, A2, B2, M1, N1)
    N2 = create_point_by_k(A2, B2, M2, k1)
    return M2, N2


def create_next_point(A1, B1, A2, B2, M1, N1, M2):
    k1 = calculate_k(A1, B1, A2, B2, M1, N1)
    N2 = create_point_by_k(A2, B2, M2, k1)
    return N2


def create_frame_0(list_data_frame):
    right_hand_result = []
    left_hand_result = []
    body_result = [[0, 0, 0]] * 33
    start_frame = list_data_frame[0]
    point_12 = create_noise_point(start_frame[2][12], 10)
    k_12_14 =  random.uniform(0.8, 1.2)
    #print ("k_body=", k_12_14)
    point_14 = create_point_by_k(start_frame[2][12], start_frame[2][14], point_12, k_12_14)
    point_14 = create_noise_point(point_14, EPSILON_BODY)
    point_16 = create_point_by_k(start_frame[2][14], start_frame[1][0], point_14, k_12_14)
    point_16 = create_noise_point(point_16, EPSILON_BODY)
    k_finger = k_12_14
    point_1_r = create_point_by_k(start_frame[1][0], start_frame[1][1], point_16, k_finger)
    point_1_r = create_noise_point(point_1_r, EPSILON_HAND_FINGER)
    right_hand_result.append(point_16)
    right_hand_result.append(point_1_r)
    for i in range(2, 21):
        if i % 4 != 1:
            point_r = create_point_by_k(start_frame[1][i - 1], start_frame[1][i],
                                        right_hand_result[len(right_hand_result) - 1], k_finger)
            point_r = create_noise_point(point_r, EPSILON_HAND_FINGER)
            right_hand_result.append(point_r)
        else:
            if i == 5:
                point_5_r = create_point_by_k(start_frame[1][0], start_frame[1][5], right_hand_result[0], k_finger)
                point_5_r = create_noise_point(point_5_r, EPSILON_HAND_FINGER)
                right_hand_result.append(point_5_r)
            else:
                point_r = create_point_by_k(start_frame[1][i - 4], start_frame[1][i], right_hand_result[i - 4],
                                            k_finger)
                point_r = create_noise_point(point_r, EPSILON_HAND_FINGER)
                right_hand_result.append(point_r)

    point_11 = create_point_by_k(start_frame[2][12], start_frame[2][11], point_12, k_12_14)
    point_11 = create_noise_point(point_11, EPSILON_BODY)
    point_13 = create_point_by_k(start_frame[2][11], start_frame[2][13], point_11, k_12_14)
    point_13 = create_noise_point(point_13, EPSILON_BODY)
    point_15 = create_point_by_k(start_frame[2][13], start_frame[0][0], point_13, k_12_14)
    point_15 = create_noise_point(point_15, EPSILON_BODY)
    point_1_l = create_point_by_k(start_frame[0][0], start_frame[0][1], point_15, k_finger)
    point_1_l = create_noise_point(point_1_l, EPSILON_HAND_FINGER)
    left_hand_result.append(point_15)
    left_hand_result.append(point_1_l)
    for i in range(2, 21):
        if i % 4 != 1:
            point_l = create_point_by_k(start_frame[0][i - 1], start_frame[0][i],
                                        left_hand_result[len(left_hand_result) - 1], k_finger)
            point_l = create_noise_point(point_l, EPSILON_HAND_FINGER)
            left_hand_result.append(point_l)
        else:
            if i == 5:
                point_5_l = create_point_by_k(start_frame[0][0], start_frame[0][5], left_hand_result[0], k_finger)
                point_5_l = create_noise_point(point_5_l, EPSILON_HAND_FINGER)
                left_hand_result.append(point_5_l)
            else:
                point_l = create_point_by_k(start_frame[0][i - 4], start_frame[0][i], left_hand_result[i - 4],
                                            k_finger)
                point_l = create_noise_point(point_l, EPSILON_HAND_FINGER)
                left_hand_result.append(point_l)

    point_24 = create_point_by_k(start_frame[2][12], start_frame[2][24], point_12, k_12_14)
    point_23 = create_point_by_k(start_frame[2][24], start_frame[2][23], point_24, k_12_14)
    point_10 = create_point_by_k(start_frame[2][12], start_frame[2][10], point_12, k_12_14)
    point_9 = create_point_by_k(start_frame[2][10], start_frame[2][9], point_10, k_finger)
    point_0 = create_point_by_k(start_frame[2][10], start_frame[2][0], point_10, k_finger)
    point_1 = create_point_by_k(start_frame[2][0], start_frame[2][1], point_0, k_finger)
    point_2 = create_point_by_k(start_frame[2][1], start_frame[2][2], point_1, k_finger)
    point_3 = create_point_by_k(start_frame[2][2], start_frame[2][3], point_2, k_finger)
    point_7 = create_point_by_k(start_frame[2][3], start_frame[2][7], point_3, k_finger)
    point_4 = create_point_by_k(start_frame[2][0], start_frame[2][4], point_0, k_finger)
    point_5 = create_point_by_k(start_frame[2][4], start_frame[2][5], point_4, k_finger)
    point_6 = create_point_by_k(start_frame[2][5], start_frame[2][6], point_5, k_finger)
    point_8 = create_point_by_k(start_frame[2][6], start_frame[2][8], point_6, k_finger)
    list_body_points = [point_0, point_1, point_2, point_3, point_4, point_5, point_6, point_7, point_8, point_9,
                        point_10, point_11,
                        point_12, point_13, point_14, point_15, point_16]
    for i in range(0, 17):
        if i <10:
            body_result[i] = create_noise_point(list_body_points[i], EPSILON_EYE)
        else:
            body_result[i] = create_noise_point(list_body_points[i], EPSILON_BODY)
    body_result[23] = create_noise_point(point_23, EPSILON_BODY)
    body_result[24] = create_noise_point(point_24, EPSILON_BODY)
    frame_0 = [left_hand_result, right_hand_result, body_result]
    return frame_0




def create_frame_t(t, list_data, frame_start):
    body_result = [[0, 0, 0]] * 33
    right_hand_result = []
    left_hand_result = []
    frame_current = list_data[t]  # A2B2
    frame_0 = list_data[0]  # A1B1
    point_12, point_14 = calculate_start_point(frame_0[2][12], frame_0[2][14], frame_current[2][12],
                                               frame_current[2][14], frame_start[2][12], frame_start[2][14])
    point_16 = create_next_point(frame_0[2][14], frame_0[1][0], frame_current[2][14], frame_current[1][0],
                                 frame_start[2][14], frame_start[1][0], point_14)
    point_11 = create_next_point(frame_0[2][12], frame_0[2][11], frame_current[2][12], frame_current[2][11],
                                 frame_start[2][12], frame_start[2][11], point_12)
    point_13 = create_next_point(frame_0[2][11], frame_0[2][13], frame_current[2][11], frame_current[2][13],
                                 frame_start[2][11], frame_start[2][13], point_11)
    point_15 = create_next_point(frame_0[2][13], frame_0[0][0], frame_current[2][13], frame_current[0][0],
                                 frame_start[2][13], frame_start[0][0], point_13)
    point_24 = create_next_point(frame_0[2][12], frame_0[2][24], frame_current[2][12], frame_current[2][24],
                                 frame_start[2][12], frame_start[2][24], point_12)
    point_23 = create_next_point(frame_0[2][24], frame_0[2][23], frame_current[2][24], frame_current[2][23],
                                 frame_start[2][24], frame_start[2][23], point_24)
    point_10 = create_next_point(frame_0[2][12], frame_0[2][10], frame_current[2][12], frame_current[2][10],
                                 frame_start[2][12], frame_start[2][10], point_12)
    point_9 = create_next_point(frame_0[2][10], frame_0[2][9], frame_current[2][10], frame_current[2][9],
                                frame_start[2][10], frame_start[2][9], point_10)
    point_0 = create_next_point(frame_0[2][10], frame_0[2][0], frame_current[2][10], frame_current[2][0],
                                frame_start[2][10], frame_start[2][0], point_10)
    point_1 = create_next_point(frame_0[2][0], frame_0[2][1], frame_current[2][0], frame_current[2][1],
                                frame_start[2][0], frame_start[2][1], point_0)
    point_2 = create_next_point(frame_0[2][1], frame_0[2][2], frame_current[2][1], frame_current[2][2],
                                frame_start[2][1], frame_start[2][2], point_1)
    point_3 = create_next_point(frame_0[2][2], frame_0[2][3], frame_current[2][2], frame_current[2][3],
                                frame_start[2][2], frame_start[2][3], point_2)
    point_4 = create_next_point(frame_0[2][0], frame_0[2][4], frame_current[2][0], frame_current[2][4],
                                frame_start[2][0], frame_start[2][4], point_0)
    point_5 = create_next_point(frame_0[2][4], frame_0[2][5], frame_current[2][4], frame_current[2][5],
                                frame_start[2][4], frame_start[2][5], point_4)
    point_6 = create_next_point(frame_0[2][5], frame_0[2][6], frame_current[2][5], frame_current[2][6],
                                frame_start[2][5], frame_start[2][6], point_5)
    point_7 = create_next_point(frame_0[2][3], frame_0[2][7], frame_current[2][3], frame_current[2][7],
                                frame_start[2][3], frame_start[2][7], point_3)
    point_8 = create_next_point(frame_0[2][6], frame_0[2][8], frame_current[2][6], frame_current[2][8],
                                frame_start[2][6], frame_start[2][8], point_6)
    list_body_points=[point_0, point_1, point_2,point_3, point_4, point_5, point_6, point_7, point_8, point_9, point_10, point_11,
                      point_12, point_13, point_14, point_15, point_16]
    for i in range (0,17):
        if i<10:
            body_result[i]= create_noise_point(list_body_points[i], EPSILON_EYE)
        else:
            body_result[i] = create_noise_point(list_body_points[i], EPSILON_BODY)
    body_result[23] = create_noise_point(point_23, EPSILON_BODY)
    body_result[24] = create_noise_point(point_24, EPSILON_BODY)

    left_hand_result.append(point_15)
    right_hand_result.append(point_16)
    point_1_l = create_next_point(frame_0[0][0], frame_0[0][1], frame_current[0][0], frame_current[0][1],
                                  frame_start[0][0], frame_start[0][1], point_15)
    left_hand_result.append(point_1_l)
    for i in range(2, 21):
        if i % 4 != 1:
            point_l = create_next_point(frame_0[0][i - 1], frame_0[0][i], frame_current[0][i - 1], frame_current[0][i],
                                        frame_start[0][i - 1], frame_start[0][i],
                                        left_hand_result[len(left_hand_result) - 1])
            point_l = create_noise_point(point_l, EPSILON_HAND_FINGER)
            left_hand_result.append(point_l)
        else:
            if i == 5:
                point_5_l = create_next_point(frame_0[0][0], frame_0[0][5], frame_current[0][0], frame_current[0][5],
                                              frame_start[0][0], frame_start[0][5], left_hand_result[0])
                point_5_l = create_noise_point(point_5_l, EPSILON_HAND_FINGER)
                left_hand_result.append(point_5_l)
            else:
                point_l = create_next_point(frame_0[0][i - 4], frame_0[0][i], frame_current[0][i - 4],
                                            frame_current[0][i],
                                            frame_start[0][i - 4], frame_start[0][i], left_hand_result[i - 4])
                point_l = create_noise_point(point_l, EPSILON_HAND_FINGER)
                left_hand_result.append(point_l)

    point_1_r = create_next_point(frame_0[1][0], frame_0[1][1], frame_current[1][0], frame_current[1][1],
                                  frame_start[1][0], frame_start[1][1], point_16)
    right_hand_result.append(point_1_r)
    for i in range(2, 21):
        if i % 4 != 1:
            point_r = create_next_point(frame_0[1][i - 1], frame_0[1][i], frame_current[1][i - 1], frame_current[1][i],
                                        frame_start[1][i - 1], frame_start[1][i],
                                        right_hand_result[len(right_hand_result) - 1])
            point_r = create_noise_point(point_r, EPSILON_HAND_FINGER)
            right_hand_result.append(point_r)
        else:
            if i == 5:
                point_5_r = create_next_point(frame_0[1][0], frame_0[1][5], frame_current[1][0], frame_current[1][5],
                                              frame_start[1][0], frame_start[1][5], right_hand_result[0])
                point_5_r = create_noise_point(point_5_r, EPSILON_HAND_FINGER)
                right_hand_result.append(point_5_r)
            else:
                point_r = create_next_point(frame_0[1][i - 4], frame_0[1][i], frame_current[1][i - 4],
                                            frame_current[1][i],
                                            frame_start[1][i - 4], frame_start[1][i], right_hand_result[i - 4])
                point_r = create_noise_point(point_r, EPSILON_HAND_FINGER)
                right_hand_result.append(point_r)
    return [left_hand_result, right_hand_result, body_result]
