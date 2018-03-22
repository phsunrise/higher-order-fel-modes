import numpy as np
import keras
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, \
        BatchNormalization, Flatten, Conv2D, AveragePooling2D, \
        MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, \
        Reshape, UpSampling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
import keras.backend as K
import h5py
from keras.callbacks import ModelCheckpoint
K.set_image_data_format('channels_last')

K.set_learning_phase(1)

def AutoEncoderNet(input_shape = (100,100,1), classes = 32):
    """
    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    
    # encoding
    X = Reshape((np.prod(input_shape),))(X_input)
    X = Dense(1024, activation='relu', name='fc1', \
              kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name='bn1')(X)
    X = Dense(256, activation='relu', name='fc2', \
              kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name='bn2')(X)
    X = Dense(128, activation='relu', name='fc3', \
              kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name='bn3')(X)
    X = Dense(64, activation='relu', name='fc4', \
              kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name='bn4')(X)
    X = Dense(classes, activation='softmax', name='code', \
              kernel_initializer = glorot_uniform(seed=0))(X) # code: length limited to 'classes'
    X = BatchNormalization(name='bn5')(X)

    # decoding
    X = Dense(64, activation='relu', name='fc6', \
              kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name='bn6')(X)
    X = Dense(128, activation='relu', name='fc7', \
              kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name='bn7')(X)
    X = Dense(256, activation='relu', name='fc8', \
              kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name='bn8')(X)
    X = Dense(1024, activation='relu', name='fc9', \
              kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name='bn9')(X)
    X = Dense(np.prod(input_shape), activation='sigmoid', name='fc10', \
              kernel_initializer = glorot_uniform(seed=0))(X)
    X = Reshape(input_shape)(X)

    # Create model
    model = Model(inputs = X_input, outputs = X, name='AutoEncoder')

    return model

## load data; data is normalized by maximum pixel value
filename = "data_combined_allruns_normalized.h5" 
with h5py.File(filename,'r') as f:
    X_train = f['imgs'].value

input_shape = (np.shape(X_train)[1], np.shape(X_train)[2], 1)

model = AutoEncoderNet(input_shape = input_shape, classes = 32)
model.compile(optimizer='adam', loss='binary_crossentropy')

print ("number of training examples = " + str(X_train.shape[0]))
print ("X_train shape: " + str(X_train.shape))

# class to record loss history
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

checkpoint = ModelCheckpoint(\
    'autoencoder_results/weights.{epoch:04d}-{val_loss:.2f}.h5', \
    save_weights_only=False)
history = LossHistory()
model.fit(X_train, X_train, epochs = 1, \
          validation_split = 0.01, batch_size = 64, \
          callbacks = [history,checkpoint])

# save model
model.save("autoencoder_results/classifier.h5")

# save history for analysis
np.savez("autoencoder_results/history.npz", losses=history.losses, \
         accuracy=history.accuracy, val_losses=history.val_losses, \
         val_accuracy=history.val_accuracy)
