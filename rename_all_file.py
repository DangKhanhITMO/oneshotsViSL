import os

def rename_files_in_folder(folder_path):
    file_list = os.listdir(folder_path)
    file_list.sort() 

    for i, file_name in enumerate(file_list):
        file_extension = os.path.splitext(file_name)[1]  # Lấy phần mở rộng của file
        new_file_name = f"{i}{file_extension+'new'}"  # Đặt tên mới cho file
        # Tạo đường dẫn đầy đủ đến file cũ và mới
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_file_name)
        os.rename(old_file_path, new_file_path)
    file_list = os.listdir(folder_path)
    file_list.sort()  
    for i, file_name in enumerate(file_list):
        file_extension = '.npy'  
        new_file_name = f"{i}{file_extension}" 
        # Tạo đường dẫn đầy đủ đến file cũ và mới
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_file_name)
        os.rename(old_file_path, new_file_path)
    print("rename done!")
def list_subdirectories(directory_path):
    subdirectories = []

    for root, dirs, files in os.walk(directory_path):
        for dir_name in dirs:
            subdirectory_path = os.path.join(root, dir_name)
            subdirectories.append(subdirectory_path)

    return subdirectories

# Thay đổi 'directory_path' thành đường dẫn thư mục gốc
directory_path = '/home/khanhlinux/ML/autodata_sign/data_standart'
subdirectories = list_subdirectories(directory_path)

# In tất cả các subdirector
for subdirectory in subdirectories:
    rename_files_in_folder(subdirectory)
