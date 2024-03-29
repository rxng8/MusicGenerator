# %%

#--------------------------------- Importing library ---------------------------------#

# OS, IO
from scipy.io import wavfile
import os, sys, shutil

# Sound Processing library
import librosa
from pydub import AudioSegment

# Midi Processing library
from mido import MidiFile, MidiTrack, Message, MetaMessage
from mido import tick2second, second2tick

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
import sklearn

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
SEQUENCE_LENGTH = 16

'''
- 4 beats is a whole note
- 2 beats is a half note
- 1 beats is a quarter note
- 0.5 beats is a eighth note
- 0.25 beats is a sixteenth note
- 0.125 beats is a thirty-two note
'''
BEATS = [4, 2, 1, 0.5, 0.25, 0.125] 


testFile = 'dataset_piano_jazz/AHouseis.mid'
midi = MidiFile(testFile)
tpb = midi.ticks_per_beat


# %%


# %%

stred, maxTick = structurize_track(midi.tracks[0], tpb)

# %%
intervals = []
for note in stred:
    intervals.append(note.duration / tpb)


# %%

intervals


# %%

intervals[-10:]

# %%

plt.bar(np.arange(len(intervals)), intervals)


# %%

for fname in os.listdir(DATA_FOLDER_PATH):
    midi = MidiFile(os.path.join(DATA_FOLDER_PATH, fname))
    if midi.type == 0:
        print(midi.tracks)
            

# %%

isinstance(midi.tracks[0][0], MetaMessage)
# %%

__find_note_form(1388, 96)

# %%

# len(str(midi.tracks[0][10]).split(" ")) != NOTE_ATTRIBUTES

# # %%

# len(str(midi.tracks[0][10]).split(" "))
# # %%
# 'channel' not in str(midi.tracks[0][10])

# # %%

# midi.tracks

# findNoteDuration(69, midi.tracks[0][3:])
# # %%

# def print_trackk(track):
#     '''
#     Do Something


#     '''
#     for i, msg in enumerate(track):
#         print(msg)
#     return

# # %%

# for i, msg in enumerate(a.tracks[1]):
    
#     print(msg)
# # %%

# tpb = midi.ticks_per_beat
# tempo = midi.tracks[0][2].tempo
# time = midi.tracks[1][2].time

# %%

#------------ TEST ----------------#
cnt = 0
tempo = 120
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



# %%

# nann = np.empty((2,10))
# nann[:] = np.nan
# # %%

# nann
# # %%
# max_line = 200

# for i in midi.play():
#     print(i)
#     if max_line == 0:
#         break
#     max_line -= 1 
# # %%

# plt.figure(figsize=(14,6))
# plt.plot(ticks[:100], notes[:100])
# plt.show()

# # %%
# plt.figure(figsize=(30,8))
# plt.scatter(times[:100], notes[:100])
# %%
# Constants for instrument creation
PERC = True
INST = False
#------------------------------------------- Note Class ---------------------------------------------------#


class NoteEvent:
    """
    NoteEvent is a fairly direct representation of Haskell Euterpea's MEvent type,
    which is for event-style reasoning much like a piano roll representation.
    start_time is absolute time for a tempo of 120bpm.
    """
    def __init__(self, start_tick, pitch, duration, noteType, velocity=64, extended=False):
        self.start_tick = start_tick  # current time
        self.pitch = pitch
        self.duration = duration
        self.stop_tick = start_tick + self.duration  # Stop time
        self.extended = extended
        self.noteType = noteType

    def data_repr(self):
        return "p" + str(self.pitch) + "t" + str(self.noteType)

    def __str__(self):
        return "NoteEvent(start_tick: {0}, duration: {1}, type: {2}, stop_tick: {3},  pitch: {4}, extended_note: {5})".format(str(self.start_tick), str(self.duration), str(self.noteType), str(self.stop_tick), str(self.pitch), str(self.extended))

    def __repr__(self):
        return str(self)

class RestEvent:
    """
    
    """
    def __init__(self):
        return

# %%

#------------------------------ Data Preprocessing ---------------------------------#


'''
Scan through a list of MIDI events looking for a matching note-off.
A note-on of the same pitch will also count to end the current note,
assuming an instrument can't play the same note twice simultaneously.
If no note-off is found, the end of the track is used to truncate
the current note.

Adding one more case: Channel is mixed within tracks
:param pitch:
:param events:
:return:
'''

def __find_note_duration(pitch, channel, events):

    sumTicks = 0
    for e in events:
        if isinstance(e, MetaMessage) or len(str(e).split(" ")) != NOTE_ATTRIBUTES or 'channel' not in str(e):
            continue
        #sumTicks = sumTicks + e.tick
        sumTicks = sumTicks + e.time
        #c = e.__class__.__name__
        c = e.type
        #if c == "NoteOffEvent" or c == "NoteOnEvent":
        if e.channel == channel and (c == "note_on" or c == "note_off"):
            if e.note == pitch:
                return sumTicks
    return sumTicks

'''
Find the interval of the notes
:param note: (NoteEvent) A particular note
:param tempo: (Integer) Tempo of the midi file
:param ticks_per_beat: (Integer) tpb
:return : (list) A list that have the same shape as constant BEATS in which define how many notes at each note_type
'''
def __find_note_form(duration, ticks_per_beat):
    
    n_beat = duration / ticks_per_beat
    # Now the beat will have the form of float denoting how many beats that note have

    beat_matrix = []
    for beat_type in BEATS:
        if n_beat >= beat_type:
            n_notes = int(n_beat // beat_type)
            beat_matrix.append(n_notes)
            n_beat -= n_notes * beat_type
        else:
            beat_matrix.append(0)

    return beat_matrix


def getChannel(track):
    '''
    Determine the channel assigned to a track.
    ASSUMPTION: all events in the track should have the same channel.
    :param track:
    :return:
    '''
    if len(track) > 0:
        e = track[0]
        #if (e.__class__.__name__ == "EndOfTrackEvent"): # mido has no end of track?
        #    return -1
        if track[0].type == 'note_on' or track[0].type=='note_off':
            return track[0].channel
    return -1

'''

'''
def structurize_track(midi_track, ticks_per_beat, default_patch=-1):


    currChannel = -1
    currTick = 0
    currPatch = default_patch

    max_stop_tick = -np.Infinity

    stred = []

    for i, msg in enumerate(midi_track):
        print(i, ": ", midi_track, ": ", msg)
        _type = msg.type

        if isinstance(msg, MetaMessage) or len(str(msg).split(" ")) != NOTE_ATTRIBUTES or 'channel' not in str(msg):
            continue

        currChannel = msg.channel
        currTick += msg.time

        if _type == 'program_change':
            currPatch = msg.program
        elif _type == 'control_change':
            pass
        elif _type == 'note_on' and msg.velocity != 0:

            # Finding durtation of the note!
            tick_duration = __find_note_duration(msg.note, currChannel, midi_track[(i+1):])

            """            # With each note in note type, append notes to the matrix
            for i, n_notes in enumerate(__find_note_form(tick_duration, ticks_per_beat)):
                event = NoteEvent(currTick, msg.note, tick_duration, i, msg.velocity, True)
                stred.append(event)

            # Since the last note of the crushendo is out, extended = False
            stred[-1].extended = False
            """

            


        elif _type == 'time_signature':
            print("TO-DO: handle time signature event")
            pass
        elif _type == 'key_signature':
            print("TO-DO: handle key signature event")
            pass # need to handle this later
        elif _type == 'note_off' or 'end_of_track' or msg.velocity == 0:
            pass # nothing to do here; note offs and track ends are handled in on-off matching in other cases.
        else:
            print("Unsupported event type (ignored): ", msg.type, vars(msg),msg)
            pass
    return stred, currTick

def map_note_to_graph(stred):
    '''
    Take a structured array of notes and map to graph
    :param stred:
    :return x:
    :return y:
    '''
    x = []
    y = []
    for i, e in enumerate(stred):
        x = np.append(x, [e.pitch, e.pitch, np.nan])
        y = np.append(y, [e.eTime, e.eTime + e.duration, np.nan])
    return x, y


'''
Create a 2D array out of a structures file!
:param stred: array of NoteEvent
:return array: a 2D array with shape (max_tick, note_range) where x axis is pitch and y axis is time.
'''
def map_note_to_array(stred, max_tick):

    array = np.zeros(shape=(max_tick, MIDI_NOTES.shape[0]))

    for i, event in enumerate(stred):
        eTime = event.eTime
        sTime = event.sTime
        pitch = event.pitch - MIDI_PITCH_TO_INDEX_CONSTANT
        velocity = event.velocity
        array[eTime:sTime, pitch] += velocity

    return array.tolist()[500:1500]

'''
Create a 2D array out of a structures file!
:param stred: array of NoteEvent
:return array: a 3D array with shape (n_notes, n_intervals, n_pitches) where axis 0 is the sequence of notes, 
    axis 1 is the type of each note(eg. quarter note, half note, whole note, etc), 
    and axis 2 is the pitch of that note.
'''
def map_note_to_sequence(stred):

    sequence = []

    note_matrix_length = MIDI_NOTES.shape[0] * len(BEATS)

    for _note in stred:
        note_matrix = np.zeros(shape=(len(BEATS), MIDI_NOTES.shape[0]))

        note_matrix[_note.noteType][_note.pitch - MIDI_PITCH_TO_INDEX_CONSTANT] = 1

        note_matrix = note_matrix.reshape(note_matrix_length)

        sequence.append(note_matrix.tolist())
        
    return np.asarray(sequence)

def generate_x_y (sequence, sequence_length):

    n_samples = int(len(sequence) // sequence_length)

    # Split the whole sequence in to n_samples of sequence length sequences.
    return_x = np.reshape(sequence[:n_samples*sequence_length], (n_samples, sequence_length, sequence.shape[1]))

    # Create training label, which is the note after a sequence at n-th sample.
    return_y = []
    for i, sample in enumerate(return_x):
        # Append the first note in the next sequence
        # Because there are no next note in after the last sequence, I just dont use the last sequence
        if i + 1 < return_x.shape[0]:
            return_y.append(return_x[i+1,0])

    return return_x[:-1], np.asarray(return_y)

"""
:param strutured_notes: (1D list) Each element is a NoteEvents.
sequence_length: Integer. 
    a length of each batch.
:return:
tuple
    (Preprocessed x, preprocess y, tokenizer x, vocab_length)
"""
def preprocess_data(strutured_notes, sequence_length=SEQUENCE_LENGTH):

    text_token_x, tokenizer = tokenize(strutured_notes)

    vocab_length = len(set(text_token_x))

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

    return preprocess_x[:-1], np.asarray(preprocess_y), tokenizer, vocab_length


"""
:param strutured_notes: (1D list) Each element is a NoteEvents.
:return: (ndarray) 1D array of associated value 
"""
def tokenize(strutured_notes):

    notes_matrix = np.asarray(strutured_notes)

    for i, note in enumerate(notes_matrix):
        notes_matrix[i] = note.data_repr()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(notes_matrix)

    sequences = np.asarray(tokenizer.texts_to_sequences(notes_matrix))
    sequences = np.reshape(sequences, (sequences.shape[0]))

    return sequences, tokenizer

# %%




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
    tensor = Embedding(18, output_dim=10, input_length=sequence_length)(in_tensor)
    tensor = LSTM(128, activation='relu', return_sequences=True)(tensor)
    tensor = Dropout(0.2)(tensor)
    tensor = LSTM(64, activation='relu')(tensor)
    tensor = Dropout(0.2)(tensor)
    tensor = Dense(vocab_length, activation='sigmoid')(tensor)

    model = Model(in_tensor, tensor)
    rmsprop = RMSprop(lr=10e-4)
    model.compile(optimizer=rmsprop, loss='sparse_categorical_crossentropy', metrics=['acc'])
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
    epochs=200,
    verbose=1
    # callbacks=callbacks_list
)

# %%


# %%

#----------------------------------------------- Predict -------------------------------------#

predict = model.predict(x[1:2])

# %%

predict[0]


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




# decoded_imgs = autoencoder.predict(x_test)

# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # display reconstruction
#     ax = plt.subplot(2, n, i + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()





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


# %%
#--------------------------------------------------Test--------------------------------------------------------#

from keras.layers import LSTM, TimeDistributed
# Dummy LSTM model

def dummyModel():
    in_tensor = Input (shape=(None, 5))
    tensor = LSTM(128, activation='relu')(in_tensor)
    tensor = Dense(1, activation='sigmoid')(tensor)
    
    model = Model(in_tensor, tensor)

    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# %%

model = dummyModel()
model.summary()
# %%

def train_generator():
    while True:
        sequence_length = np.random.randint(10, 100)
        x_train = np.random.random((1000, sequence_length, 5))
        # y_train will depend on past 5 timesteps of x
        y_train = x_train[:,0,0]
        y_train = to_categorical(y_train)
        yield x_train, y_train
# %%
model.fit_generator(train_generator(), steps_per_epoch=30, epochs=10, verbose=1)

# %%

for x,y in train_generator():
    print(x.shape)

