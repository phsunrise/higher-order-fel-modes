import numpy as np
np.random.seed(0)
import keras
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
import keras.backend as K
import h5py
from keras.callbacks import ModelCheckpoint
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

def SimpleNet(input_shape = (100,100,1), classes = 1):
    """
    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    
    # Stage 1
    X = Conv2D(16, (3, 3), strides = (1, 1), name = 'conv1', padding = 'same', \
        kernel_initializer = glorot_uniform(seed=0))(X_input) # now 100 x 100 x 16
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X) # now 50 x 50 x 16

    # Stage 2
    X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv2', padding = 'same', \
        kernel_initializer = glorot_uniform(seed=0))(X) # now 50 x 50 x 32
    X = BatchNormalization(axis = 3, name = 'bn_conv2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X) # now 25 x 25 x 32

    # Stage 3
    X = Conv2D(64, (3, 3), strides = (1, 1), name='conv3', padding = 'same', \
        kernel_initializer=glorot_uniform(seed=0))(X) # now 25 x 25 x 64
    X = BatchNormalization(axis = 3, name = 'bn_conv3')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X) # now 12 x 12 x 64
    
    # Stage 4
    X = Conv2D(128, (3, 3), strides = (1, 1), name='conv4', padding = 'same', \
        kernel_initializer=glorot_uniform(seed=0))(X) # now 12 x 12 x 128 
    X = BatchNormalization(axis = 3, name = 'bn_conv4')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(3, 3))(X) # now 4 x 4 x 128 

    # Stage 5
    X = Conv2D(256, (2, 2), strides = (1, 1), name='conv5', padding = 'same', \
        kernel_initializer=glorot_uniform(seed=0))(X) # now 4 x 4 x 256
    X = BatchNormalization(axis = 3, name = 'bn_conv5')(X)
    X = Activation('relu')(X)

    # final stage
    X = GlobalAveragePooling2D()(X) # now 1 x 1 x 256

    X = Dense(classes, activation='sigmoid', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='SimpleNet')

    return model

#load data

filename = "data_combined.h5"
with h5py.File(filename,'r') as f:
    X_train = f['train_X'].value
    Y_train = f['train_Y'].value
    X_test = f['test_X'].value
    Y_test = f['test_Y'].value

classes = Y_train.max()
input_shape = (np.shape(X_train)[1],np.shape(X_train)[2],1)

model = SimpleNet(input_shape = input_shape, classes = classes)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Normalize image vectors let's see whether we need that
#X_train = X_train_orig/255.
#X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
#Y_train = convert_to_one_hot(Y_train_orig, 6).T
#Y_test = convert_to_one_hot(Y_test_orig, 6).T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []
        self.val_losses = []
        self.val_accuracy = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))
    def on_epoch_end(self, batch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        self.val_accuracy.append(logs.get('val_acc'))

checkpoint = ModelCheckpoint('./weights.{epoch:04d}-{val_loss:.2f}.hdf5', save_weights_only=False)
history = LossHistory()
model.fit(X_train, Y_train, epochs = 50, \
        validation_split = 0.1, batch_size = 32, callbacks = [history,checkpoint])
preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
model.save("classifier.h5")

# save history
np.savez("history.npz", losses=history.losses, accuracy=history.accuracy, \
         val_losses=history.val_losses, val_accuracy=history.val_accuracy)
