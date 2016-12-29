import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import csv
from keras.models import Sequential, Model
from keras.layers import Conv2D, ConvLSTM2D, Dense, MaxPooling2D, Dropout, Flatten, Reshape, merge, Input
from keras.optimizers import Adam

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('image_dir', 'DrivingData/IMG/', 'Simulator Image data')
flags.DEFINE_string('data_path', 'DrivingData/driving_log.csv', 'Simulator CSV')
flags.DEFINE_float('learn_rate', 0.0001, 'Trainign learning rate')

print ('Init completed')

with open(FLAGS.data_path, 'r') as f:
    reader = csv.reader(f)
    csv = np.array([row for row in reader])
# [:,0] - center image
# [:,1] - left image
# [:,2] - right image
# [:,3] - steering (-0.7, 0.438)
# [:,4] - throttle (0, 1)
# [:,5] - brake (0, 1)
# [:,6] - speed (0, 9.8)

# Process single image
def proc_img(img): # input is 160x320x3
    img = img[59:138:2, 0:-1:2, :] # select vertical region and take each second pixel to reduce image dimensions
    img = (img / 127.5) - 1.0 # normalize colors from 0-255 to -1.0 to 1.0
    return img # return 40x160x3 image

# Read image names and remove IMG/ prefix
#image_names_center = np.array(csv[:,0]) # do not use it in this model
image_names_left = np.array(csv[:,1])
image_names_right = np.array(csv[:,2])
image_names_full = np.concatenate((image_names_left, image_names_right))
# read steering data and apply adjustment for left / right images
y_data = np.array(csv[:,3], dtype=float)
y_data_left = y_data+0.08
y_data_right = y_data-0.08
y_data_full = np.concatenate((y_data_left, y_data_right))
image_data = np.zeros((len(image_names_full),40, 160, 3),dtype=float)
for i in range(len(image_names_full)):
    image_name = image_names_full[i][4:]
    image_names_full[i] = image_name
    image = mpimg.imread(FLAGS.image_dir+image_name)
    image_data[i] = proc_img(image)

print ('Image data read and processed')

# Random sort for data and split test and validation sets
def newRandomTestValidationSplit(X, y):
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.01, random_state=111)
    return X_tr, X_val, y_tr, y_val


inp = Input(shape=(40,160,3))
x = Conv2D(3, 1, 1, border_mode='same', activation='relu')(inp)
x = MaxPooling2D((2,2))(x) #20x80
x1 = Conv2D(32, 3, 3, border_mode='same', activation='relu')(x)
x1 = Conv2D(32, 3, 3, border_mode='same', activation='relu')(x1)
x1 = MaxPooling2D((2,2))(x1) #20x80 #10x40
x1 = Dropout(0.5)(x1)
flat1 = Flatten()(x1)

x2 = Conv2D(64, 3, 3, border_mode='same', activation='relu')(x1)
x2 = Conv2D(64, 3, 3, border_mode='same', activation='relu')(x2)
x2 = MaxPooling2D((2,2))(x2) #10x40 #5x20
x2 = Dropout(0.5)(x2)
flat2 = Flatten()(x2)

x3 = Conv2D(64, 3, 3, border_mode='same', activation='relu')(x2)
x3 = Conv2D(64, 3, 3, border_mode='same', activation='relu')(x3)
x3 = MaxPooling2D((2,2))(x3) #5x20 # 3x10
x3 = Dropout(0.5)(x3)
flat3 = Flatten()(x3)

x4 = merge([flat1, flat2, flat3], mode='concat')
x5 = Dense(512, activation='relu')(x4)
x6 = Dense(128, activation='relu')(x5)
x7 = Dense(16, activation='relu')(x6)
out = Dense(1, activation='linear')(x7)

model = Model(input=inp, output=out)
#model.summary()

# Compile, train and save
model.compile(optimizer=Adam(lr=FLAGS.learn_rate), loss='mse')

print ('Start training')

X_tr, X_val, y_tr, y_val = newRandomTestValidationSplit(image_data, y_data_full)
history = model.fit(X_tr, y_tr, batch_size=64, nb_epoch=5, verbose=1, validation_data=(X_val, y_val))

json = model.to_json()
model.save_weights('model.h5')
with open('model.json', 'w') as f:
    f.write(json)

print ('Model saved')
