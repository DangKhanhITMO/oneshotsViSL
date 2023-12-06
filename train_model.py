import os

import keras
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from keras.layers import Dropout,BatchNormalization
path_data_train= '/home/khanhlinux/ML/autodata_sign (copy)/Data/0512/train'
path_data_test= '/home/khanhlinux/ML/autodata_sign (copy)/Data/0512/test'
path_to_log_train= "/home/khanhlinux/ML/autodata_sign (copy)/Data/0512/log/log_train.txt"
path_to_log_test= "/home/khanhlinux/ML/autodata_sign (copy)/Data/0512/log/log_test.txt"
path_to_X_train ='/home/khanhlinux/ML/autodata_sign (copy)/Data/0512/datanumpy/X_train.npy'
path_to_y_train ='/home/khanhlinux/ML/autodata_sign (copy)/Data/0512/datanumpy/y_train.npy'
path_to_X_test ='/home/khanhlinux/ML/autodata_sign (copy)/Data/0512/datanumpy/X_test.npy'
path_to_y_test = '/home/khanhlinux/ML/autodata_sign (copy)/Data/0512/datanumpy/y_test.npy'


class MyThresholdCallback( keras.callbacks.Callback):
    def __init__(self, threshold, count):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold
        self.count = count

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs["categorical_accuracy"]
        vad_acc=logs["val_categorical_accuracy"]
        if val_acc >= self.threshold and vad_acc>=self.threshold:
          self.count += 1
          if self.count >=10:
            self.model.stop_training = True
        else:
          self.count = 0
my_callback = MyThresholdCallback(threshold=0.95,count=0)
def list_categoria(rootdir):
    list_categorical = []
    for sub_dir in os.listdir(rootdir):
        d = os.path.join(rootdir, sub_dir)
        if os.path.isdir(d):
            list_categorical.append(sub_dir)
    return (list_categorical)
# lay danh sach cac file trong thu muc
def list_file_dir(root_dir):
  return os.listdir(root_dir)


list_actions= list_categoria(path_data_train)
array_categoria = np.array(list_actions)
action_map = {label: num for num, label in enumerate(array_categoria)}
try:
    f = open('/home/khanhlinux/ML/autodata_sign (copy)/Data/0512/log/categoria.txt', "w")
    f.write(str(action_map))
    f.close()
except:
    print('error write log at ')
def check_lable_list():
  list_labels_test= list_categoria(path_data_test)
  for label in list_labels_test:
    if label not in list_actions:
      print( label)
      return False
  return True

def get_data(path_to_data, len_sequence, path_to_log):
  X, y = [], []
  i = 0
  if check_lable_list:
    list_sub_dir = list_actions
    for sub_dir in list_sub_dir:
      print(sub_dir)
      path_to_file = os.path.join(path_to_data, sub_dir)
      list_files = list_file_dir(path_to_file)
      for name_file in list_files:
        window=[]
        temp_y = np.array([0]* len(list_actions))
        full_path= os.path.join(path_to_file, name_file)
        if os.path.isfile(full_path):
          wd = np.load(full_path, allow_pickle = True)
          for index in range(len_sequence):
            frame_i= np.concatenate(np.array(wd[index]), axis=None)
            window.append(frame_i)
          temp_y[action_map[sub_dir]]=1
          X.append(window)
          y.append(temp_y)
          try:
            f = open(path_to_log, "a")
            f.write(str(i)+ ': ' + str(full_path) +'   '+ str(action_map[sub_dir] ) + '  '+ str(sub_dir)+ '\n' )
            i +=1
            f.close()
          except:
            print('error write log at ' + str(i)+' '+ str(full_path))
  return X, y
def prerosecc_data(path_to_X, path_to_y):
    index1= [ i for i in range(126,225)]
    index2 = [i for i in range(126) if i%3== 2]
    X = np.load(path_to_X)
    y = np.load(path_to_y)
    result = []
    for i in range(X.shape[0]):
        window = []
        for j in range(X.shape[1]):
            temp_X1= np.delete(X[i][j],index1,0)
            temp_X2= np.delete(temp_X1,index2,0)
            window.append(temp_X2)
        result.append(window)
    return np.array(result), y

X1_train, y1_train = get_data(path_data_train, 20, path_to_log_train)
X1_test, y1_test = get_data(path_data_test, 20, path_to_log_test)
np.save( path_to_X_train, X1_train)
np.save(path_to_y_train, y1_train)
np.save(path_to_X_test, X1_test)
np.save(path_to_y_test, y1_test)
X_train,y_train = prerosecc_data(path_to_X_train, path_to_y_train)
X_test, y_test  = prerosecc_data(path_to_X_test, path_to_y_test)
print(X_train.shape)
print(y_train.shape)
model = Sequential()
model.add(GRU(128, return_sequences=True, activation='relu', input_shape=(20, 84)))
model.add(Dropout(0.2))
model.add(GRU(128, return_sequences=True, activation='relu', input_shape=(20, 84)))
model.add(Dropout(0.2))
model.add(GRU(256, return_sequences=False, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(y_test.shape[1], activation='softmax'))
model.summary()
'''
model = Sequential()
model.add(GRU(128, return_sequences=True, activation='relu', input_shape=(20, 84)))
model.add(Dropout(0.2))
model.add(GRU(256, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(GRU(256, return_sequences=False, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

'''
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# Huấn luyện mô hình và ghi lại các giá trị accuracy và loss
history = model.fit( X_train, y_train   , epochs=1000, validation_data=( X_test, y_test), callbacks=[my_callback], batch_size=128)
model.save('/home/khanhlinux/ML/autodata_sign (copy)/Data/0512/model/sin.keras')
# Lấy các giá trị accuracy và loss từ lịch sử huấn luyện
train_acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
# Vẽ biểu đồ accuracy
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Vẽ biểu đồ loss
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
# print(x_test.shape)
# print(my_X.shape)'''



