import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Declare a global variable to hold the time from previous sample/image.
t_prev = 0.
# Declare width offset. This is the assumed offset in meters for
# the left and right camera mounting from the center camera.
# Note: This value is assumed to be 0.8 m based on intuition.
width_offset = 0.8

# Define a function to extract the time from the image name.
# Returns: The time in seconds.
def getTimeFromName(source):
    split_path = source.split('/')[-1]
    t = split_path.split('_')
    tend = t[-1].split('.')
    ts = float(t[6]) + float(tend[0])/1000

    return ts

# Define a function to compute the correction factor.
# The correction factor is calculated based on the formula 'v = rw',
# where, v is the ego vehicle's velocity and r is the perpendicular
# distance to each of the cameras and w is the rotational velocity.
# Returns: The correction factor used for offsetting the left and right images.
def getCorrectionFactor(t_current, speed_str):
    global t_prev

    # Transform the speed to SI units.
    speed = float(speed_str) * 0.44704
    # Compute the delta time.
    # Note: The dt calculated in the first sample will have an offset as t_prev
    # is initialized to zero.
    dt = abs(t_current - t_prev)

    # Manage the wrapping of seconds.
    if dt > 10.:
        dt = abs(dt - 60.)

    cf = (speed / width_offset) * dt
    t_prev = t_current

    return cf

print("Reading CSV files ...")
# Read from csv file
data = []
# Since generators are used in this method, it is necessary to
# calculate the correction factor before any shuffling is performed.
# Here, the correction factor is calculated for each image and
# appended as an element to the read line as list.
with open('./linux_sim/driving_log.csv') as f:
    reader = csv.reader(f)

    for line in reader:
        t_current = getTimeFromName(line[0])
        cf = getCorrectionFactor(t_current, line[6])
        line.append(str(cf))
        data.append(line)

# Split the available data into train and validation sets.
# 20% of the available data is used for validation.
train_samples, validation_samples = train_test_split(data, test_size=0.2)

print("Reading done ...")

# Define a generator to return features and labels as a batch for each call.
# Note: Augmented data is created and input into each batch.
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates.
        sklearn.utils.shuffle(samples) # Shuffles for each epoch.
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # Center image
                center = batch_sample[0]
                center_image = plt.imread(center)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                # Correction factor
                cf = float(batch_sample[7])

                # Left image with compensated correction factor.
                left = batch_sample[1]
                left_image = plt.imread(left)
                left_angle = center_angle + cf
                images.append(left_image)
                angles.append(left_angle)

                # Right image with compensated correction factor.
                right = batch_sample[2]
                right_image = plt.imread(right)
                right_angle = center_angle - cf
                images.append(right_image)
                angles.append(right_angle)

                # Prepare and add augmented images.
                # Flipped image center
                flip_center = cv2.flip(center_image, flipCode=1)
                images.append(flip_center)
                angles.append(-center_angle)

                # Flipped image left
                flip_left = cv2.flip(left_image, flipCode=1)
                images.append(flip_left)
                angles.append(-left_angle)

                # Flipped image center
                flip_right = cv2.flip(right_image, flipCode=1)
                images.append(flip_right)
                angles.append(-right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

print("Running model")

# Define the model architecture.
# Note: The model architecture used here is an adapted version from
# Ford's own real-time lane monitoring system using CNN published on IEEE.
# It is worth mentioning that the sizes of the images used by Ford were
# very different. Hence the right filter size for convolutions were tuned.
model = Sequential()
# Crop the images using the keras to allow parallel processing on GPU.
# Results in significant performance increase while done on a GPU.
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
# Layer 1
# Convolution with ELU for activation.
# Normalization
# Max pooling.
model.add(Convolution2D(32, (10, 10), strides=6, activation='elu'))
model.add(Lambda(lambda x: (x/127.5) - 1.))
model.add(MaxPooling2D(2, 2))
# Layer 2
# Convolution with ELU for activation.
# Normalization
# Max pooling.
model.add(Convolution2D(64, (5, 5), strides=1, activation='elu'))
model.add(Lambda(lambda x: (x/127.5) - 1.))
model.add(MaxPooling2D(2, 2))
# Flatten for fully connected layers.
model.add(Flatten())
# Layer 3
# Fully connected layer with 50% dropout of ELU activations.
model.add(Dense(2048, activation='elu'))
model.add(Dropout(0.5))
# Layer 4
# Fully connected layer with 50% dropout of ELU activations.
model.add(Dense(317, activation='elu'))
model.add(Dropout(0.5))
# Layer 5
# Fully connected layer with 50% dropout of ELU activations.
# Note: This is an extra layer added (Not included in Ford's document) for correct stepping down in neurons.
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
# Layer 6
# Ends in one neuron representing the steering angle.
model.add(Dense(1))
# Compile the model.
model.compile(loss='mse', optimizer='adam')

# Define the batch size.
batchsize = 32

# Define the train and validation generators.
train_generator = generator(train_samples, batch_size=batchsize)
validation_generator = generator(validation_samples, batch_size=batchsize)

# Train the model with the generators for 6 epochs.
# Keras 2 was used for building and training the model.
# Some parameters' name differ on Keras 2 from Keras 1.
# samples_per_epoch was changed to steps_per_epoch and takes the number of batches as input.
# nb_val_samples was changed to validation_steps and takes the number of batches as input.
# use_multiprocessing is set to true to allow the usage of CPU together with GPU for generator usage.
history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples)/batchsize,
                                     validation_data=validation_generator, use_multiprocessing=True,
                                    validation_steps=len(validation_samples)/batchsize, nb_epoch=6, verbose=2)

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# Save the model.
print("Saving model ...")
model.save('model.h5')
print("Model saved.")
