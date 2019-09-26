# %%

#--------------------------------- Importing library ---------------------------------#

# OS, IO
from scipy.io import wavfile
import os, sys, shutil

# Sound Processing library
import librosa
from pydub import AudioSegment

# Midi Processing library
from mido import MidiFile
from mido import Message, MetaMessage
from mido import tick2second, second2tick

# Math Library
import numpy as np

# Display library
import IPython.display as ipd
import matplotlib.pyplot as plt
%matplotlib inline
plt.interactive(True)
import librosa.display

# Data Preprocessing
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
import sklearn

# Deep Learning Library
from keras.models import Model, Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, LSTM, Bidirectional, GRU
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.optimizers import Adam, SGD
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.utils import Sequence
from keras.optimizers import Adam, SGD, RMSprop

# %%
#------------------------------- CONSTANTS ----------------------------------------#

atavachron = 'test_data/Atavachron.mid'

noc = 'test_data/chno0902.mid'

midi = MidiFile(noc)

# %%

#Test note off
x = midi.tracks[1]
i = 0
for msg in x:
    print(msg)
    if i >= 200:
        break
    i += 1
# %%
for msg in midi.play():
    print(msg)


# %%

tpb = midi.ticks_per_beat
tempo = midi.tracks[0][2].tempo
# time = midi.tracks[1][2].time

# %%

#------------ TEST ----------------#
cnt = 0
notes = []
times = []
curr_time = 0
for j, message in enumerate(midi.tracks[1]):
    if isinstance(message, Message) and message.type == 'note_on':
        time = message.time
        sec = tick2second(time, tpb, tempo)
        notes.append(message.note)
        curr_time += sec
        times.append(curr_time)


# %%

notes

# %%
max_line = 200

for i in midi.play():
    print(i)
    if max_line == 0:
        break
    max_line -= 1 
# %%

plt.figure(figsize=(14,6))
plt.plot(ticks[:100], notes[:100])
plt.show()

# %%
plt.figure(figsize=(30,8))
plt.scatter(times[:100], notes[:100])

# %%

class DataPreprocessor:

# %%

class DataGenerator(Sequence):
    """
    :param data_path: (String) This is the base folder data.
    :param batch_size: (int32) This is the base folder data.
    :param dim: (Tuple: (a, b, c)) 3D tuple shape of input dimension
    :param n_channels: (int32) Number of channel.
    :param n_classes: (int32) Number of classes.
    :param shuffle: (boolean) Specify whether or not you want to shuffle the data to be trained.
    """
    def __init__(self, data_path, batch_size=32, dim=(128,1308), n_channels=1,
             n_classes=10, shuffle=True, validation_split=0.1):
        """
        :var self.classes:
        :var self.labels:
        :var self.fname:
        :var self.data:
        :var self.dim:
        :var self.batch_size:
        :var self.list_IDs:
        :var self.n_channels:
        :var self.n_classes:
        :var self.shuffle:
        :var self.tokenizer:
        :var self.data_path:
        """
        self.classes = []
        self.labels = []
        self.fname = []
        self.data = []
        self.data_validation = []

        self.validation_split = validation_split
        self.data_size = 0
        self.data_shape = (None,None)
        self.data_path = data_path
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = []
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.load_data()
        
    """
    :param data_path: (String) The actual base folder of data
    """
    def load_data(self):

        
    """
    Utilities method for classes
    :param filename: Name of the file
    """
    def load_midifile(filename):


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        return

    def __len__(self):
        return

    def __getitem__(self, index):
        return



# %%

x.tracks[11][30]
# %%

isinstance(x.tracks[1][300], MetaMessage)


