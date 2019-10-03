# Music Generator Project

This is a written project with regard to processing sound using keras deep learning library. These are the planned application:
- [x] [Genres classification](https://github.com/dewanderelex/SoundProcessing)
- [ ] Music auto-tagging
- [ ] Instrumental classification
- [ ] Tempo classification
- [x] [Deep Learning Jazz generator](https://github.com/dewanderelex/MusicGenerator)
- [ ] Sort music by genres, artist, tempo, tags
- [ ] Piano analysis
-------------
## Music Generator Project using deep learning neural network



-------------
## Comission Note:

#### Sep 22, 2019:
- Researching on processing Midi file format: tracks, messages, channels, etc.

#### Sep 23, 2019:
- More research on processing Midi file format: tracks, messages, channels, etc.

#### Sep 26, 2019:
- More research on MIDI file processing and feature engineering.
- Add data generator.
- Add code to read and parse midi file, converting them to structrured form.
- Add helper and utilization function.

#### Sep 28, 2019:
- Edit reference for paper
- Add critical data preprocessing function: convert array of NoteEvent to a 2D array with shape (max_tick, note_range) where x axis is pitch and y axis is time.
- Update compressed dataset

#### Sep 29, 2019:
- Finish Data Preprocessor and Data Generator. Working on building Convolutional Neural Network with analytic pixel. The paper can be found [here](https://arxiv.org/pdf/1701.05517.pdf)
- Try building model with generative adversarial neural network.

#### Sep 30, 2019:
- Build model with PixelCNN++ philosophy.
- Build model with Variational Autoencoder philosophy.
- Train 16000 epochs on CNN model, with dataset of single-track midi files.
- Log file surpassing (edit .gitignore)

#### Oct 1, 2019:
- Switch from CNN model to seg-to-seg LSTM model. Do more research.

#### Oct 3, 2019:
- Split the main notebook into one with music generator processing as image processing, the other as sequence processing (CNN and LSTM).
- Utilize a unique class for a note event for efficient data pre-processing.
- Add function to build note matrix from midi tracks.

-------------


## Author: Alex Nguyen
#### Gettysburg College Class of 2022
