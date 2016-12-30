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
# center image, left image, right image
# steering (-0.8, 0.8)
# throttle (0, 1)
# brake (0, 1)
# speed (0, 9.8)

# Process single image
def proc_img(img): # input is 160x320x3
    img = img[59:138:2, 0:-1:2, :] # select vertical region and take each second pixel to reduce image dimensions
    img = (img / 127.5) - 1.0 # normalize colors from 0-255 to -1.0 to 1.0
    return img # return 40x160x3 image

# Read image names and remove IMG/ prefix
#image_names_center = np.array(csv[:,0]) # not used it in this model
image_names_left = np.array(csv[:,1])
image_names_right = np.array(csv[:,2])
image_names_full = np.concatenate((image_names_left, image_names_right))
# read steering data and apply adjustment for left / right images
y_data = np.array(csv[:,3], dtype=float)
y_data_left = y_data+0.08
y_data_right = y_data-0.08
y_data_full = np.concatenate((y_data_left, y_data_right))
print ('CSV loaded')

# Random sort for data and split test and validation sets
def newRandomTestValidationSplit(X, y):
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.01, random_state=111)
    return X_tr, X_val, y_tr, y_val

# Batch generator for training data
def generate_image_batch_tr(names, y_data, batch_size = 32):
    total_items = len(names)
    curr_item = 0
    while (True):
        image_data = np.zeros((batch_size,40, 160, 3),dtype=float)
        steering_data = np.zeros((batch_size),dtype=float)
        for j in range(batch_size):
            image_name = names[curr_item][4:]
            image = mpimg.imread(FLAGS.image_dir+image_name)
            image_data[j] = proc_img(image)
            steering_data[j] = y_data[curr_item]
            curr_item = (curr_item+1)%total_items
        yield image_data, steering_data

# Batch generator for validation data (in this implementation same as for training data)
def generate_image_batch(names, y_data, batch_size = 32):
    total_items = len(names)
    curr_item = 0
    while (True):
        image_data = np.zeros((batch_size,40, 160, 3),dtype=float)
        steering_data = np.zeros((batch_size),dtype=float)
        for j in range(batch_size):
            image_name = names[curr_item][4:]
            image = mpimg.imread(FLAGS.image_dir+image_name)
            image_data[j] = proc_img(image)
            steering_data[j] = y_data[curr_item]
            curr_item = (curr_item+1)%total_items
        yield image_data, steering_data

# ----------------------
# Model - ideas from VG type network
inp = Input(shape=(40,160,3))
# First convolution is for model to determine the 'best' colorspace weights
x = Conv2D(3, 1, 1, border_mode='same', activation='relu')(inp)
# Reduce dimensions
x = MaxPooling2D((2,2))(x) #20x80

# First convolution layer
x1 = Conv2D(32, 3, 3, border_mode='same', activation='relu')(x)
x1 = Conv2D(32, 3, 3, border_mode='same', activation='relu')(x1)
x1 = MaxPooling2D((2,2))(x1) #10x40
x1 = Dropout(0.5)(x1)
flat1 = Flatten()(x1) # Used for the merge before first fully connected layer

# Second convolution layer
x2 = Conv2D(64, 3, 3, border_mode='same', activation='relu')(x1)
x2 = Conv2D(64, 3, 3, border_mode='same', activation='relu')(x2)
x2 = MaxPooling2D((2,2))(x2) #5x20
x2 = Dropout(0.5)(x2)
flat2 = Flatten()(x2) # Used for the merge before first fully connected layer

# Second convolution layer
x3 = Conv2D(64, 3, 3, border_mode='same', activation='relu')(x2)
x3 = Conv2D(64, 3, 3, border_mode='same', activation='relu')(x3)
x3 = MaxPooling2D((2,2))(x3) #2x10
x3 = Dropout(0.5)(x3)
flat3 = Flatten()(x3) # Used for the merge before first fully connected layer

# Merge the flattened ouputs after each convolution layer
x4 = merge([flat1, flat2, flat3], mode='concat')
# Fully connected layers
x5 = Dense(512, activation='relu')(x4)
x6 = Dense(128, activation='relu')(x5)
x7 = Dense(16, activation='relu')(x6)
out = Dense(1, activation='linear')(x7)

model = Model(input=inp, output=out)
model.summary()

# Compile, train and save
model.compile(optimizer=Adam(lr=FLAGS.learn_rate), loss='mse')
print ('Split data')
X_tr_names, X_val_names, y_tr, y_val = newRandomTestValidationSplit(image_names_full, y_data_full)

print ('Start training')
# Training and validation inputs are fed from generators
# Number of samples based on data_set size and adjusted to fit batch size
history = model.fit_generator(generate_image_batch_tr(X_tr_names, y_tr, 64),samples_per_epoch=15744,
                              nb_epoch=5,
                              validation_data=generate_image_batch(X_val_names, y_val, 32),
                              nb_val_samples=160)
# Save model
json = model.to_json()
model.save_weights('model.h5')
with open('model.json', 'w') as f:
    f.write(json)

print ('Model saved')