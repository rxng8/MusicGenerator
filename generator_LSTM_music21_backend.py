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
TEST_FOLDER_PATH = 'test_data'
MIDI_NOTES = np.arange(21, 109)
# MIDI_NOTES_MAP = {
#     '21': 'A0',
#     '22': 'B0',
#     # TODO: Implement!
# }
MIDI_PITCH_TO_INDEX_CONSTANT = np.min(MIDI_NOTES)
NOTE_ATTRIBUTES = 5
TICK_SCALER = 0.1
N_CHANNELS = 1
SEQUENCE_LENGTH = 8
N_FILE_TRAIN = 10


# %%

#------------------------------ Data Preprocessing ---------------------------------#

def load_data(folder_name, max_file=N_FILE_TRAIN):
    notes = []
    cnt = 0
    for _fname in os.listdir(folder_name):
        if ('.mid' not in _fname and '.MID' not in _fname):
            print ("{} file is not valid".format(_fname))
        elif (cnt < N_FILE_TRAIN):
            notes.append(load_midi(os.path.join(folder_name, _fname)))
            cnt += 1

    # vocab = sorted(set([note for note in notes]))

    return notes

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

def preprocess_data_one_hot(notes, sequence_length=SEQUENCE_LENGTH):

    '''
    :param strutured_notes: (1D list) Each element is a NoteEvents.
    sequence_length: Integer. 
        a length of each batch.
    :return:
    tuple
        (Preprocessed x, preprocess y, tokenizer x, vocab_length)
    '''

    # tmp is 1d representation of notes
    tmp = [note for midi_data in notes for note in midi_data]
    # text_token_x, tokenizer = tokenize(tmp)
    # vocab = sorted(set([note for note in text_token_x]))
    vocab = sorted(set([note for note in tmp]))
    note_to_idx = dict([(name, i) for i, name in enumerate(vocab)])


    preprocess_x = np.zeros(shape=(0, sequence_length, len(vocab)))
    preprocess_y = np.zeros(shape=(0, len(vocab)))

    for midi_data in notes:

        midi_data_np = np.asarray(midi_data)
        n_samples = int(midi_data_np.shape[0] // sequence_length)

        # Split the whole sequence into n_samples of sequence length sequences.
        single_preprocess_x = np.reshape(midi_data_np[:n_samples*sequence_length], (n_samples, sequence_length))

        single_preprocess_x_after = np.zeros(shape=(single_preprocess_x.shape[0], single_preprocess_x.shape[1], len(vocab)))

        for i in range(single_preprocess_x.shape[0]):
            for j in range(single_preprocess_x.shape[1]):
                single_preprocess_x_after[i][j][note_to_idx[single_preprocess_x[i][j]]] = 1
        # Now preprocess_x_after will have shape (n_batch, seq_len, one-hot-coded vector)

        # print(np.asarray(single_preprocess_x_after.tolist()[:-1]).shape)
        # print(preprocess_x)

        preprocess_x = np.append(preprocess_x, single_preprocess_x_after[:-1], axis=0)
        # preprocess_x.append(single_preprocess_x_after.tolist()[:-1])

        # print(np.asarray(preprocess_x).shape)

        # Create training label, which is the note after a sequence at n-th sample.
        single_preprocess_y = []
        for i, sample in enumerate(single_preprocess_x_after):
            # Append the first note in the next sequence
            # Because there are no next note in after the last sequence, I just dont use the last sequence
            if i + 1 < single_preprocess_x_after.shape[0]:
                single_preprocess_y.append(single_preprocess_x_after[i+1,0])

        preprocess_y = np.append(preprocess_y, single_preprocess_y, axis=0)

    # return np.asarray(preprocess_x), np.asarray(preprocess_y), tokenizer
    return preprocess_x, preprocess_y, vocab, note_to_idx

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
#-------------------------------------- Test -------------------------------#

tmp = np.asarray([[0,0,0,1],
        [0,0,1,0],
        [0,0,0,1],
        [1,0,0,0]])

tmp = tmp.reshape((1, tmp.shape[0], tmp.shape[1]))

arr = np.append(tmp, tmp, axis=0)
arr = np.append(arr, tmp, axis=0)
y_arr = np.asarray([[0,0,0,1],
            [0,0,0,1],
            [0,0,0,1]])

# %%

arr.shape
# %%

y_arr.shape
# %%

#------------------------------------- Runner ----------------------------------------#

arr = load_data(TEST_FOLDER_PATH)


# %%

arr[0]

# %%


x, y, vocab, note_to_idx = preprocess_data_one_hot(arr)
vocab_length = len(vocab)

# %%

x.shape

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
    tensor = Embedding(vocab_length, output_dim=100, input_length=sequence_length)(in_tensor)
    tensor = LSTM(128, activation='relu', return_sequences=True)(tensor)
    tensor = Dropout(0.2)(tensor)
    tensor = LSTM(64, activation='relu')(tensor)
    tensor = Dropout(0.2)(tensor)
    tensor = Dense(vocab_length, activation='softmax')(tensor)

    model = Model(in_tensor, tensor)
    rmsprop = RMSprop(lr=10e-5)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['acc'])
    return model

def lstm_model(vocab_length, sequence_length=SEQUENCE_LENGTH):
    in_tensor = Input(shape=(sequence_length, vocab_length))
    tensor = LSTM(128, activation='relu', return_sequences=True)(in_tensor)
    tensor = Dropout(0.2)(tensor)
    tensor = LSTM(64, activation='relu')(tensor)
    tensor = Dropout(0.2)(tensor)
    tensor = Dense(vocab_length, activation='softmax')(tensor)

    model = Model(in_tensor, tensor)
    rmsprop = RMSprop(lr=10e-5)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['acc'])
    return model

def dummy_lstm ():
    in_tensor = Input(shape=(4,4))
    tensor = LSTM(16, activation='relu', return_sequences=True)(in_tensor)
    tensor = Dropout(0.2)(tensor)
    tensor = LSTM(8, activation='relu')(tensor)
    tensor = Dropout(0.2)(tensor)
    tensor = Dense(4, activation='softmax')(tensor)

    model = Model(in_tensor, tensor)
    rmsprop = RMSprop(lr=10e-4)
    model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['acc'])
    return model

# %%

########################################### RUNNER ####################################

model = lstm_model(vocab_length)
model.summary()

# %%

########################################### RUNNER ####################################

model = simple_lstm_model(vocab_length)
model.summary()


# %%

# filepath = 'checkpoint/'
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# callbacks_list = [checkpoint]

history = model.fit(
    x=x,
    y=y,
    batch_size=8,
    epochs=2000
    # callbacks=callbacks_list
)

# %%

#-------------------------- Training history Analysis ------------------------------#
import pydot
from keras.utils import plot_model
plot_model(model, to_file='model.png')
# %%

from IPython.display import SVG
from keras.utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))

# %%


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()


# %%

# Save the entire model
model.save('Bach_prelude_and_fuge_in_C_major_BWV_846_model.h5')


# %%

# Save the weights
model.save_weights('Bach_prelude_and_fuge_in_C_major_BWV_846_model_weights.h5')

# Save the model architecture
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())


# %%

from keras.models import model_from_json

# Model reconstruction from JSON file
with open('model_architecture.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('model_weights.h5')

# %%

np.argmax(model.predict(x[:1])[0])

# %%
backward_dict = dict()
for note in note_to_idx.keys():
    index = note_to_idx[note]
    backward_dict[index] = note

# pick a random sequence from the input as a starting point for the prediction
n = np.random.randint(0, x.shape[0] - 1)
sequence = x[n]
start_sequence = sequence.reshape(1, SEQUENCE_LENGTH, vocab_length)
output = []

# Let's generate a song of 100 notes
for i in range(0, 100):
    newNote = model.predict(start_sequence, verbose=0)
    # Get the position with the highest probability
    index = np.argmax(newNote)
    encoded_note = np.zeros((vocab_length))
    encoded_note[index] = 1
    output.append(encoded_note)
    sequence = start_sequence[0][1:]
    start_sequence = np.concatenate((sequence, encoded_note.reshape(1, vocab_length)))
    start_sequence = start_sequence.reshape(1, SEQUENCE_LENGTH, vocab_length)
    

# Now output is populated with notes in their string form
for element in output:
    print(element)


# %%

finalNotes = [] 
for element in output:
    index = list(element).index(1)
    finalNotes.append(backward_dict[index])
    
offset = 0
output_notes = []
    
# %%

x[0,:10,:]


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

