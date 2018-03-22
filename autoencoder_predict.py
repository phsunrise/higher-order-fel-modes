import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, load_model
import keras.backend as K
import h5py

plt.ion()
K.set_learning_phase(0)

## Load labeled data (which contains a higher percentage of higher-order images)
f = h5py.File("data_combined_normalized.h5", 'r')
X = np.array(f.get("train_X").value)

## Get output images using trained model
model = load_model("autoencoder_results/classifier.h5")
Y = model.predict(X)

## Get encoder part of the model
encoder = Model(inputs=model.input, \
             outputs=model.get_layer('code').output)
## Get the "codes"
encoder_output = encoder.predict(X)

## Get decoder part of the model 
decoder = K.function([model.get_layer('code').output], \
                     [model.output])

## Visualize each component of the "code", by setting the component 
## to 1 and the rest to 0, and using it as the input for decoder
classes = 32 
plt.figure(figsize=(16,4)) 
for i in range(classes):
    code = np.zeros((1, classes)).astype(float)
    code[0, i] = 1.
    decoder_output = decoder([code])[0]
    plt.subplot(4,8,i+1)
    plt.title('label = {}'.format(i))
    plt.imshow(decoder_output.reshape(100, 100),interpolation = 'nearest')
    plt.axis('off')

## Randomly pick input and output images to see if the output captures
## important features of the input
for i in range(6):
    fig, axes = plt.subplots(3, 8, figsize=(30, 5))
    for i_ax in range(8):
        ind = np.random.randint(len(X))
        ax = axes[0][i_ax]
        im1 = ax.imshow(X[ind].reshape(100,100))
        fig.colorbar(im1, ax=ax)
        ax.set_title(str(ind))
        ax = axes[1][i_ax]
        im2 = ax.imshow(Y[ind].reshape(100,100))
        fig.colorbar(im2, ax=ax)
        ax = axes[2][i_ax]
        ax.plot(encoder_output[ind], 'ro')
        print "code: ", encoder_output[i]

f.close()
