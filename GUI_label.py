## original code came from: http://physicalmodelingwithpython.blogspot.com/2016/04/make-your-own-gui-with-python.html
try:
    # This will work in Python 2.7
    import Tkinter
except ImportError:
    # This will work in Python 3.5
    import tkinter as Tkinter
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
import os

h5_filename = "run181_binned_all.h5"
data_dir = "/reg/data/ana13/xpp/xppx22715/results/imgs/data_files/"

f = h5py.File(data_dir + h5_filename, 'r')
imgs = f.get("imgs")

labels_filename = h5_filename[:-3] + "_labels.npy" 
labels_init = -1  # initial value for labels
if os.path.isfile(labels_filename):
    # copy existing label file to backup
    _ifile = 0
    while os.path.isfile(labels_filename + "_backup_%d"%_ifile):
        _ifile += 1
    os.system("cp %s %s_backup_%d" % (labels_filename, labels_filename, _ifile))
    # load existing label
    print "Reading existing labels from %s" % labels_filename
    labels = np.load(labels_filename).astype(int)
else:
    labels = np.zeros(imgs.shape[0]) + labels_init

num_list = [] # list of indices

### START GUI setup ###
# Create main application window.
root = Tkinter.Tk()


# Create a text box explaining the application.
greeting = Tkinter.Label(text="display images", \
                    font=('Arial', '20', 'bold'))
greeting.pack(side='top')

# Create a frame for variable names and entry boxes for their values.
frame = Tkinter.Frame(root)
frame.pack(side='top')

# Variables for the calculation
num_input = Tkinter.StringVar()
label_input = Tkinter.StringVar()

# Create text boxes and entry boxes for the variables.
# Use grid geometry manager instead of packing the entries in.
# image number
num_text = Tkinter.Label(frame, text='Image #:') 
num_text.grid(row=0, column=0)

num_entry = Tkinter.Entry(frame, width=8, textvariable=num_input)
num_entry.grid(row=0, column=1)

# label
label_text = Tkinter.Label(frame, text='Label:') 
label_text.grid(row=1, column=0)

label_entry = Tkinter.Entry(frame, width=8, textvariable=label_input)
label_entry.grid(row=1, column=1)

# initialization of GUI
img_plot = None
def GUI_init():
    global num_input, label_input, img_plot, labels
    # display first unlabelled image 
    num = 0
    while num < len(labels):
        if labels[num] != labels_init:
            num += 1 
        else:
            label_input.set('0')
            break
    if num >= len(labels):
        print "reached last image"
        label_input.set(str(labels[-1]))
        num = len(labels) - 1

    num_input.set(str(num))

    fig, ax = plt.subplots(1, 1)
    img_plot = ax.imshow(imgs[num])
    plt.show()

# initialization: execute after 10ms
root.after(10, GUI_init)

def next_img():
    global num_input, label_input, labels, img_plot
    num = int(num_input.get())
    while num < len(labels):
        if labels[num] != labels_init:
            num += 1 
        else:
            break
    if num >= len(labels):
        print "reached last image"
        num = len(labels) - 1

    num_input.set(str(num))
    label_input.set('0')
    img_plot.set_data(imgs[num])
    plt.draw()

def save_label(event=None):
    global label_input, num_input, labels
    label = int(label_input.get())
    num = int(num_input.get())
    labels[num] = label
    num_list.append(num)
    next_img() # go to next image

def last_img():
    global label_input, num_input, labels, num_list
    num = num_list.pop()
    num_input.set(str(num))
    label_input.set(str(labels[num]))

def save_to_file():
    global labels, labels_filename 
    np.save(labels_filename, labels)
    print "Saved to %s" % labels_filename

def exit_GUI(event=None):
    global root 
    plt.close()
    root.destroy()
    

save_label_button = Tkinter.Button(root, command=save_label, text="Save and next")
save_label_button.grid(row=1, column=2)
save_label_button.pack(fill='y')

last_button = Tkinter.Button(root, command=last_img, text="Last image")
last_button.grid(row=2, column=0)
last_button.pack(fill='y')

save_button = Tkinter.Button(root, command=save_to_file, text="Save to file")
save_button.grid(row=2, column=1)
save_button.pack(fill='y')

exit_button = Tkinter.Button(root, command=exit_GUI, text="Exit")
exit_button.grid(row=2, column=2)
exit_button.pack(fill='y')


# Allow pressing <Return> to save label.
root.bind('<Return>', save_label)

# Allow pressing <Esc> to close the window.
root.bind('<Escape>', exit_GUI)

### END GUI setup ###

# Activate the window.
root.mainloop()
