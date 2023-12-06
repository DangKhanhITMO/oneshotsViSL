import numpy as np
import create_point as p
import os
import MediaPipeProcess.create_numpy_data as test
from MediaPipeProcess.keypoint_extract import extract_landmarks_by_img
def get_files_in_directory(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_list.append(file_path)
    return file_list

#temp= test.get_list_frame('/home/khanhlinux/ML/autodata_sign (copy)/Data/0412/videos/9. Train/MVI_3110.MOV')
#test.view_data(temp)
Y = np.load("/home/khanhlinux/ML/autodata_sign (copy)/Data/0412/test/10. Plane/17.npy", allow_pickle=True)
print(Y)
'''result = []
list_img = get_files_in_directory('/home/khanhlinux/ML_NN/mediaPipe/image_signer')
frame_0 = extract_landmarks_by_img(list_img[2])
print(list_img[2])
result.append(Y[0])
for i in range(1, 20):
    frame_i = p.create_frame_t(i, Y, frame_0)
    result.append(frame_i)
print(result[1][2])
print("----------------------")
print(Y[1][2])'''
test.view_data(Y)
