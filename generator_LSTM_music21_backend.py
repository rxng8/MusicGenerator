# %%

#--------------------------------- Importing library ---------------------------------#

# OS, IO
from scipy.io import wavfile
import os, sys, shutil

# Sound Processing library
import librosa
from pydub import AudioSegment

# Midi Processing library
from music21.midi import MidiFile, MidiTrack, MidiEvent, MidiException, DeltaTime
from music21 import converter, instrument, note, chord, midi, stream

# Math Library
import numpy as np
import random

# Display library
import IPython.display as ipd
import matplotlib.pyplot as plt
# %matplotlib inline
plt.interactive(True)
import librosa.display

# Data Preprocessing
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

# Deep Learning Library
from keras.models import Model, Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, LSTM, Bidirectional, GRU
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D
from keras.layers import Embedding
from keras.optimizers import Adam, SGD
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.utils import Sequence
from keras.optimizers import Adam, SGD, RMSprop, Adadelta
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

# Utils



# %%
#------------------------------- CONSTANTS ----------------------------------------#

ata = 'test_data/Atavachron.mid'
noc = 'test_data/chno0902.mid'
noo = 'test_data/NoOneInTheWorld.mid'

# Time Signature
NUMERATOR = 4
DENOMNMINATOR = 4

DATA_FOLDER_PATH = 'dataset_piano_jazz'
MIDI_NOTES = np.arange(21, 109)
MIDI_NOTES_MAP = {
    '21': 'A0',
    '22': 'B0',
    # TODO: Implement!
}
MIDI_PITCH_TO_INDEX_CONSTANT = np.min(MIDI_NOTES)
NOTE_ATTRIBUTES = 5
TICK_SCALER = 0.1
N_CHANNELS = 1
SEQUENCE_LENGTH = 4
N_FILE_TRAIN = 40


testFile = 'dataset_piano_jazz/AHouseis.mid'
# midi_data = MidiFile()
# midi_data.open(testFile)
# midi_data.read()

# %%

x = converter.parse(testFile)

# %%

x[0].flat.notes[:50]

# %%
strings = "abc"

strings[0] == 'a'

# %%

#------------------------------ Data Preprocessing ---------------------------------#

def load_data(folder_name, max_file=N_FILE_TRAIN):
    notes = []
    cnt = 0
    for _fname in os.listdir(folder_name):
        if ('.mid' not in _fname and '.MID' not in _fname):
            print ("{} file is not valid".format(_fname))
        elif (cnt < 10):
            notes = np.append(notes, load_midi(os.path.join(folder_name, _fname)))
            cnt += 1

    vocab = sorted(set([note for note in notes]))

    return notes, vocab

def load_midi(file_name):
    '''
    Take one midi file, load it into a chord array
    :param file_name: (String) Path to file
    :return: (ndarray) 1D array with shape (max_sequences) of notes and chords.
    :return: (ndarray) 1D array, set of unique notes and chords.
    '''

    midi_data = converter.parse(file_name)
    
    notes = []

    for element in midi_data.flat.notes:
        if (isinstance(element, note.Note)):
            rep = str(element.pitch)
            if ('#' in rep or '-' in rep):
                tmp = ""
                for char in rep:
                    if char == '#':
                        tmp += 's'
                    elif char == '-':
                        tmp += 'f'
                    else:
                        tmp += char
                notes.append(tmp)
            else:
                notes.append(rep)
            
        elif (isinstance(element, chord.Chord)):
            notes.append('c'.join(str(n) for n in element.normalOrder))

    print("Loaded {} file".format(file_name))

    return notes


def preprocess_data(notes, sequence_length=SEQUENCE_LENGTH):

    '''
    :param strutured_notes: (1D list) Each element is a NoteEvents.
    sequence_length: Integer. 
        a length of each batch.
    :return:
    tuple
        (Preprocessed x, preprocess y, tokenizer x, vocab_length)
    '''

    text_token_x, tokenizer = tokenize(notes)

    n_samples = int(text_token_x.shape[0] // sequence_length)

    # Split the whole sequence in to n_samples of sequence length sequences.
    preprocess_x = np.reshape(text_token_x[:n_samples*sequence_length], (n_samples, sequence_length))

    # Create training label, which is the note after a sequence at n-th sample.
    preprocess_y = []
    for i, sample in enumerate(preprocess_x):
        # Append the first note in the next sequence
        # Because there are no next note in after the last sequence, I just dont use the last sequence
        if i + 1 < preprocess_x.shape[0]:
            preprocess_y.append(preprocess_x[i+1,0])

    preprocess_y = to_categorical(preprocess_y, num_classes=len(vocab))

    return preprocess_x[:-1], np.asarray(preprocess_y), tokenizer


"""
:param strutured_notes: (1D list) Each element is a NoteEvents.
:return: (ndarray) 1D array of associated value 
"""
def tokenize(notes):

    notes_matrix = [note for note in np.asarray(notes)]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(notes_matrix)

    sequences = np.asarray(tokenizer.texts_to_sequences(notes_matrix))
    sequences = np.reshape(sequences, (sequences.shape[0]))

    return sequences, tokenizer

# %%

#------------------------------------- Runner ----------------------------------------#

arr, vocab = load_data(DATA_FOLDER_PATH)

x, y, tokner = preprocess_data(arr)

vocab_length = len(vocab)

# %%

len(vocab)
# %%

y.shape

# %%

#------------------------------------- Data Extraction ----------------------------------#

"""
:param data_path: (String) The actual file
"""
def extract_data(data_path):
    
    midi = MidiFile(data_path)
    tpb = midi.ticks_per_beat
    print('Extracting {}'.format(data_path))
    if midi.type == 0:
        
        st, maxTick = structurize_track(midi.tracks[0], tpb) # If type == 0 -> midi just have 1 track.
        arr = map_note_to_sequence(st)
        
    return arr


# %%

########################################### Runner ########################################

midi = MidiFile(testFile)

arr, maxTick = structurize_track(midi.tracks[0], midi.ticks_per_beat)

x, y, token, vocab_length = preprocess_data(arr)

# %%
TEST_INDEX = 11
target = x[TEST_INDEX][0]
terIdx = -1

print(arr[TEST_INDEX])

for i, data in enumerate(x):
    if data[0] == target:
        terIdx = i
        print(arr[terIdx])


# %%

#----------------------------------------- Models ----------------------------------#


####################### Model with multi-label classification ? ######################



def conv_autoencoder_1 (shape=(None, MIDI_NOTES.shape[0], N_CHANNELS)):
    in_tensor = Input(shape=shape)

    tensor = Conv2D(64, (1,1), activation = 'relu', padding='valid')(in_tensor)
    tensor = MaxPooling2D(2)(tensor)

    tensor = Conv2D(64, (1,1), activation = 'relu', padding='valid')(tensor)
    tensor = MaxPooling2D(2)(tensor)

    tensor = Conv2D(64, (1,1), activation = 'relu', padding='valid')(tensor)
    tensor = MaxPooling2D(2)(tensor)

    tensor = Conv2D(64, (1,1), activation = 'relu', padding='valid')(tensor)
    tensor = UpSampling2D(2)(tensor)

    tensor = Conv2D(64, (1,1), activation = 'relu', padding='valid')(tensor)
    tensor = UpSampling2D(2)(tensor)

    tensor = Conv2D(N_CHANNELS, (1,1), activation = 'relu', padding='valid')(tensor)
    tensor = UpSampling2D(2)(tensor)

    model = Model(in_tensor, tensor)
    adam = Adam(lr = 10e-4)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])

    return model

def conv_autoencoder_2(shape=(None, MIDI_NOTES.shape[0], N_CHANNELS)):

    in_tensor = Input(shape=shape)
    
    tensor = Conv2D(32, (3, 3), activation='relu', padding='same')(in_tensor)
    tensor = MaxPooling2D((2, 2), padding='same')(tensor)
    tensor = Conv2D(16, (3, 3), activation='relu', padding='same')(tensor)
    tensor = MaxPooling2D((2, 2), padding='same')(tensor)
    tensor = Conv2D(8, (3, 3), activation='relu', padding='same')(tensor)
    encoded = MaxPooling2D((2, 2), padding='same')(tensor)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    tensor = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    tensor = UpSampling2D((2, 2))(tensor)
    tensor = Conv2D(16, (3, 3), activation='relu', padding='same')(tensor)
    tensor = UpSampling2D((2, 2))(tensor)
    tensor = Conv2D(32, (3, 3), activation='relu', padding='same')(tensor)
    tensor = UpSampling2D((2, 2))(tensor)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(tensor)

    autoencoder = Model(in_tensor, decoded)

    adadelta = Adadelta(lr=10e-4)
    autoencoder.compile(optimizer=adadelta, loss='mean_squared_error')

    return autoencoder

def conv_vae():
    # TODO: To implement!
    return


def simple_lstm_model(vocab_length, sequence_length=SEQUENCE_LENGTH):
    in_tensor = Input(shape=(sequence_length,))
    tensor = Embedding(50, output_dim=30, input_length=sequence_length)(in_tensor)
    tensor = LSTM(128, activation='relu', return_sequences=True)(tensor)
    tensor = Dropout(0.2)(tensor)
    tensor = LSTM(64, activation='relu')(tensor)
    tensor = Dropout(0.2)(tensor)
    tensor = Dense(vocab_length, activation='softmax')(tensor)

    model = Model(in_tensor, tensor)
    rmsprop = RMSprop(lr=10e-5)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['acc'])
    return model


# %%

########################################### RUNNER ####################################

model = simple_lstm_model(vocab_length)
model.summary()


# %%

filepath = 'checkpoint/'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]

history = model.fit(
    x=x,
    y=y,
    batch_size=16,
    epochs=2000,
    verbose=1
    # callbacks=callbacks_list
)

# %%


# %%

#----------------------------------------------- Predict -------------------------------------#

predict = model.predict(x[1:2])

# %%

predict[0]
x = []

for i, data in enumerate(predict[0]):
    if data > 1e-2:
        x.append(i)

# %%

x
# %%

#---------------------------------------------- Show --------------------------------------------#

plt.imshow(x[0,:,:,0])


# %%

#------------------------------------------------- Show -------------------------------------------#

plt.imshow(predict[0,:,:,0])


# %%

x[0,:,:,0]

# %%


predict[0,:,:,0]




# %%

#---------------------------------------------------- MIDI Write file ----------------------------------#

composed = MidiFile()
composed.ticks_per_beat = 96


track = MidiTrack()

track.append(Message('note_on', note=64, velocity=64, time=0))
track.append(Message('note_on', note=62, velocity=64, time=200))
track.append(Message('note_on', note=60, velocity=64, time=200))
track.append(Message('note_off', note=64, velocity=64, time=200))
track.append(Message('note_off', note=62, velocity=64, time=200))
track.append(Message('note_off', note=60, velocity=64, time=200))

composed.tracks.append(track)

composed.save('composed.mid')


# %%

compo = MidiFile('composed.mid')

