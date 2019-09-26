def findNoteDuration(pitch, events):
    '''
    Scan through a list of MIDI events looking for a matching note-off.
    A note-on of the same pitch will also count to end the current note,
    assuming an instrument can't play the same note twice simultaneously.
    If no note-off is found, the end of the track is used to truncate
    the current note.

    Adding one more case: note-is off when velocity = 0
    :param pitch:
    :param events:
    :return:
    '''
    sumTicks = 0
    for e in events:
        #sumTicks = sumTicks + e.tick
        sumTicks = sumTicks + e.time
        #c = e.__class__.__name__
        c = e.type
        #if c == "NoteOffEvent" or c == "NoteOnEvent":
        if c == "note_on" or c == "note_off":
            if e.note == pitch:
                return sumTicks
    return sumTicks