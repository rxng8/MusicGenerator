from scipy.io import wavfile
import os, sys, shutil

# Sound Processing library
import librosa
from pydub import AudioSegment

# Midi Processing library
from mido import MidiFile
from mido import Message, MetaMessage
from mido import tick2second, second2tick

#Math
import numpy as np




def get_note_range(midi_track):
    '''
    Do somethhing

    :param midi_track:
    :return minNote:
    :return maxNote:
    '''
    minNote = np.Infinity
    maxNote = -np.Infinity
    for i, msg in enumerate(midi_track):
        if isinstance(msg, Message) and msg.type == 'note_on':
            note = msg.note
            if note > maxNote:
                maxNote = note
            if note < minNote:
                minNote = note
    return minNote, maxNote





