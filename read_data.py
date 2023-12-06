import os
import shutil
def create_directory(path, directory_name):
    # Combine the path and directory name
    directory_path = os.path.join(path, directory_name)

    try:
        # Create the directory
        os.mkdir(directory_path)
        print("Directory created successfully!")
    except FileExistsError:
        print("Directory already exists!")
    except Exception as e:
        print(f"Error: {str(e)}")


def copy_file(source_path, destination_path, file_name, new_file_name=None):
    """
    Copy a file from a source directory to a destination directory.

    Args:
        source_path (str): Path to the source directory.
        destination_path (str): Path to the destination directory.
        file_name (str): Name of the file to be copied.
        new_file_name (str, optional): New name for the copied file (default is None).
    """
    # Combine the source directory path and file name
    source_file_path = os.path.join(source_path, file_name)

    # Determine the new file name in the destination directory
    if new_file_name is None:
        new_file_name = file_name
    destination_file_path = os.path.join(destination_path, new_file_name)

    try:
        # Copy the file from source to destination
        shutil.copy2(source_file_path, destination_file_path)
        print(f"Successfully copied file {file_name}!")
    except FileNotFoundError:
        print(f"File {file_name} not found!")
    except Exception as e:
        print(f"Error: {str(e)}")
def list_files(dir_path):
    # list to store files
    res = []
    try:
        for file_path in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path, file_path)):
                res.append(file_path)
    except FileNotFoundError:
        print(f"The directory {dir_path} does not exist")
    except PermissionError:
        print(f"Permission denied to access the directory {dir_path}")
    except OSError as e:
        print(f"An OS error occurred: {e}")
    return res


def extrac_keypoint_from_videos(root_dir, des_dir):
    from MediaPipeProcess.create_numpy_data import write_data
    for sub_dir in os.listdir(root_dir):
        path = os.path.join(root_dir, sub_dir)
        if os.path.isdir(path):
            create_directory(des_dir, sub_dir)
            new_path = os.path.join(des_dir, sub_dir)
            list_file = list_files(path)
            for i in range(len(list_file)):
                path_to_file = os.path.join(path, list_file[i])
                write_data(new_path, path_to_file, str(i) + ".npy")

#main('/home/khanhlinux/ML_NN/mediaPipe/Videos','/home/khanhlinux/ML_NN/mediaPipe/LSTM/data_test')