import h5py
import numpy as np
import sys
from bin_data import rebin

data_dir = "/reg/data/ana13/xpp/xppx22715/results/imgs/data_files/"
filenames = ["run179_binned_all", "run180_binned_all", "run181_binned_all"]
train_size = [2000, 2000] # for labels 0 and 1-4
test_size = [100, 100]

# read in all labels 
for i_f, filename in enumerate(filenames):
    labelarray = np.load(filename+"_labels.npy")
    
    # get the indices
    ind0 = np.where(labelarray==0)[0]
    ind1 = np.where(np.logical_and(labelarray>=1, labelarray<=4))[0]

    try:
        # final result is a np.array of shape (N, 3)
        # each entry is (label, index in file, index of file)
        label_0 = np.vstack((label_0, np.hstack((np.ones((len(ind0),1))*0, \
            ind0.reshape(len(ind0), 1), np.ones((len(ind0),1))*i_f))))
        label_1 = np.vstack((label_1, np.hstack((np.ones((len(ind1),1))*1,\
            ind1.reshape(len(ind1), 1), np.ones((len(ind1),1))*i_f))))
    except NameError:
        label_0 = np.hstack((np.ones((len(ind0),1))*0, \
            ind0.reshape(len(ind0), 1), np.ones((len(ind0),1))*i_f))
        label_1 = np.hstack((np.ones((len(ind1),1))*1, \
            ind1.reshape(len(ind1), 1), np.ones((len(ind1),1))*i_f))

# shuffle the labels
label_0 = label_0.astype(int)
label_1 = label_1.astype(int)
print "total number label=0: %d, label=1-4: %d" % (\
            len(label_0), len(label_1))
print "now shuffling"
np.random.shuffle(label_0)
np.random.shuffle(label_1)
    
# allocate train and test sets
train_set = np.vstack((label_0[:train_size[0]], label_1[:train_size[1]]))
test_set = np.vstack((label_0[train_size[0]:train_size[0]+test_size[0]], \
                label_1[train_size[1]:train_size[1]+test_size[1]]))
np.random.shuffle(train_set)
np.random.shuffle(test_set)
print "total train size: %d, test size: %d" % (\
    len(train_set), len(test_set))
    

# stack the images
imgs = []
for filename in filenames:
    f = h5py.File(data_dir + filename + ".h5", 'r')
    imgs.append(f.get("imgs"))

# function to resize the images to 100 x 100 x 1
def resize(img):
    return rebin(img[100:400, 150:450], [100, 100]).reshape(100, 100, 1)

train_X = []
train_Y = []
for _i, (label, ind, i_file) in enumerate(train_set):
    train_X.append(resize(imgs[i_file][ind]))
    train_Y.append([label])
    if _i % 100 == 0:
        print "done %d" % _i

test_X = []
test_Y = []
for _i, (label, ind, i_file) in enumerate(test_set):
    test_X.append(resize(imgs[i_file][ind]))
    test_Y.append([label])
    if _i % 100 == 0:
        print "done %d" % _i

# save data to hdf5 file
h5filename = 'data_combined.h5'
with h5py.File(h5filename, 'w') as f:
    f.create_dataset('train_X', data = train_X)
    f.create_dataset('train_Y', data = train_Y)
    f.create_dataset('test_X', data = test_X)
    f.create_dataset('test_Y', data = test_Y)
