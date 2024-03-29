from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, Embedding
from keras.layers import Conv1D, MaxPooling1D, Flatten, Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint
from keras import backend
from keras.utils import np_utils
import os
from os.path import isfile

from timeit import default_timer as timer

mono=True


def get_class_names(path="Preproc/"):  # class names are subdirectory names in Preproc/ directory
    class_names = os.listdir(path)
    return class_names

def get_total_files(path="Preproc/",train_percentage=0.8): 
    sum_total = 0
    sum_train = 0
    sum_test = 0
    subdirs = os.listdir(path)
    for subdir in subdirs:
        files = os.listdir(path+subdir)
        n_files = len(files)
        sum_total += n_files
        n_train = int(train_percentage*n_files)
        n_test = n_files - n_train
        sum_train += n_train
        sum_test += n_test
    return sum_total, sum_train, sum_test

def get_sample_dimensions(path='Preproc/'):
    classname = os.listdir(path)[0]
    files = os.listdir(path+classname)
    infilename = files[0]
    audio_path = path + classname + '/' + infilename
    melgram = np.load(audio_path)
    print("   get_sample_dimensions: melgram.shape = ",melgram.shape)
    return melgram.shape
 

def encode_class(class_name, class_names):  # makes a "one-hot" vector for each class name called
    try:
        idx = class_names.index(class_name)
        vec = np.zeros(len(class_names))
        vec[idx] = 1
        return vec
    except ValueError:
        return None

def shuffle_XY_paths(X,Y,paths):   # generates a randomized order, keeping X&Y(&paths) together
    assert (X.shape[0] == Y.shape[0] )
    idx = np.array(range(Y.shape[0]))
    np.random.shuffle(idx)
    newX = np.copy(X)
    newY = np.copy(Y)
    newpaths = paths
    for i in range(len(idx)):
        newX[i] = X[idx[i],:,:]
        newY[i] = Y[idx[i],:]
        newpaths[i] = paths[idx[i]]
    return newX, newY, newpaths


'''
So we make the training & testing datasets here, and we do it separately.
Why not just make one big dataset, shuffle, and then split into train & test?
because we want to make sure statistics in training & testing are as similar as possible
'''
def build_datasets(train_percentage=0.8, preproc=False):
    if (preproc):
        path = "Preproc/"
    else:
        path = "Samples/"

    class_names = get_class_names(path=path)
    print("class_names = ",class_names)

    total_files, total_train, total_test = get_total_files(path=path, train_percentage=train_percentage)
    print("total files = ",total_files)

    nb_classes = len(class_names)

    # pre-allocate memory for speed (old method used np.concatenate, slow)
    mel_dims = get_sample_dimensions(path=path)  # Find out the 'shape' of each data file
    #X_train = np.zeros((total_train, mel_dims[1], mel_dims[2], mel_dims[3]))   
    X_train = np.zeros((total_train, mel_dims[1], mel_dims[2], mel_dims[3]))   
    Y_train = np.zeros((total_train, nb_classes))  
    X_test = np.zeros((total_test, mel_dims[1], mel_dims[2], mel_dims[3]))  
    Y_test = np.zeros((total_test, nb_classes))  
    paths_train = []
    paths_test = []

    train_count = 0
    test_count = 0
    for idx, classname in enumerate(class_names):
        this_Y = np.array(encode_class(classname,class_names) )
        this_Y = this_Y[np.newaxis,:]
        class_files = os.listdir(path+classname)
        n_files = len(class_files)
        n_load =  n_files
        n_train = int(train_percentage * n_load)
        printevery = 100
        print("")
        for idx2, infilename in enumerate(class_files[0:n_load]): 
            audio_path = path + classname + '/' + infilename
            if (0 == idx2 % printevery):
                print('\r Loading class: {:14s} ({:2d} of {:2d} classes)'.format(classname,idx+1,nb_classes), ", file " ,idx2+1," of ",n_load,": ",audio_path)
            #start = timer()
            if (preproc):
              melgram = np.load(audio_path)
              sr = 44100
            else:
              aud, sr = librosa.load(audio_path, mono=mono,sr=None)
              melgram = librosa.logamplitude(librosa.feature.melspectrogram(aud, sr=sr, n_mels=96),ref_power=1.0)[np.newaxis,np.newaxis,:,:]

            melgram = melgram[:,:,:,0:mel_dims[3]]   # just in case files are differnt sizes: clip to first file size
       
            #end = timer()
            #print("time = ",end - start) 
            if (idx2 < n_train):
                # concatenate is SLOW for big datasets; use pre-allocated instead
                #X_train = np.concatenate((X_train, melgram), axis=0)  
                #Y_train = np.concatenate((Y_train, this_Y), axis=0)
                X_train[train_count,:,:] = melgram
                Y_train[train_count,:] = this_Y
                paths_train.append(audio_path)     # list-appending is still fast. (??)
                train_count += 1
            else:
                
                X_test[test_count,:,:] = melgram
                Y_test[test_count,:] = this_Y
                #X_test = np.concatenate((X_test, melgram), axis=0)
                #Y_test = np.concatenate((Y_test, this_Y), axis=0)
                paths_test.append(audio_path)
                test_count += 1
        print("")

    print("Shuffling order of data...")
    X_train, Y_train, paths_train = shuffle_XY_paths(X_train, Y_train, paths_train)
    X_test, Y_test, paths_test = shuffle_XY_paths(X_test, Y_test, paths_test)

    return X_train, Y_train, paths_train, X_test, Y_test, paths_test, class_names, sr



def build_model(X,Y,nb_classes):
    nb_filters = 8  # number of convolutional filters to use
    pool_size = (1, 1)  # size of pooling area for max pooling
    kernel_size = (3,3)  # convolution kernel size
    nb_layers = 4
    input_shape = ( X.shape[0], X.shape[1],)


    Xre = X.reshape(1133,1,96*2584)

   
   ## Working LSTM
    model = Sequential()
    model.add(LSTM(12, input_shape=(1,248064)) )
    model.add(Dropout(0.25))
    model.add(ELU(alpha=0.1)) 
    model.add(Dense(8) )
    model.add(Activation('relu'))
    model.add(Dense(len(class_names)) )
    model.add( Activation("softmax" ))

    ## Working CNN
    model = Sequential()
    model.add(Conv1D(128, 1,input_shape=(1,248064)) )
    model.add(MaxPooling1D(pool_size=1, strides=None, padding='valid'))
    model.add(Dropout(0.25))
    model.add(ELU(alpha=0.1)) 
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dense(len(class_names)) )
    model.add( Activation("softmax" ))

    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
    model.summary()
    model.fit(Xre,Y_train, batch_size=10, epochs=20, validation_split=0.2)


    # main_input  =  Input(shape=input_shape ,dtype='int32',  name='main_input')
    # embedding = Embedding(input_dim=128, output_dim=128) (main_input )
    # conv = Conv1D(filters =128, kernel_size =5, padding='same', activation ='relu', strides =1) (embedding)
    # max_pool = MaxPooling1D(pool_size=2, padding ='same') (conv)
    # encode = LSTM( 64, return_sequences=False) (max_pool)
    # output  = Dense( nb_classes, activation ='sigmoid') (encode)
    # model = Model(inputs=main_input , outputs =output)

    # early_release = keras.callbacks.EarlyStopping(monitor='acc', min_delta=0, patience=5, verbose=0, mode='auto')
    # tbCallback = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs', write_graph=True)
    # Result = model.fit(trainX, trainy, epochs=55, shuffle=True, batch_size=32, verbose=1, validation_split=0.3, callbacks=[tbCallback,early_release])

    model = Sequential()

    model.add(LSTM( 128,  input_shape=input_shape,  return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM( X[2],return_sequences=False))
    model.add(Dropout(0.2))


    model = Sequential()
    model.add( Embedding(input_dim=128, output_dim=128,  input_shape = input_shape))
    #model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))
    model.add(LSTM(nb_filters))
    model.add(Activation('relu'))
    
    for layer in range(nb_layers-1):
        model.add(LSTM(nb_filters))
        model.add(BatchNormalization(axis=1))
        model.add(ELU(alpha=1.0))  
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    return model
    

if __name__ == '__main__':
    np.random.seed(1)

    # get the data
    X_train, Y_train, paths_train, X_test, Y_test, paths_test, class_names, sr = build_datasets(preproc=True)

    # make the model
    model = build_model(X_train,Y_train, nb_classes=len(class_names))
    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
    model.summary()

    # Initialize weights using checkpoint if it exists. (Checkpointing requires h5py)
    load_checkpoint = True
    checkpoint_filepath = 'weights.hdf5'
    if (load_checkpoint):
        print("Looking for previous weights...")
        if ( isfile(checkpoint_filepath) ):
            print ('Checkpoint file detected. Loading weights.')
            model.load_weights(checkpoint_filepath)
        else:
            print ('No checkpoint file detected.  Starting from scratch.')
    else:
        print('Starting from scratch (no checkpoint)')
    checkpointer = ModelCheckpoint(filepath=checkpoint_filepath, verbose=1, save_best_only=True)


    # train and score the model
    batch_size = 10
    nb_epoch = 10
    print("Batch size is ",batch_size,".\nNumber of epochs ",nb_epoch)
    model.fit(X_train, Y_train, batch_size=batch_size,  epochs=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test), callbacks=[checkpointer])
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    save = input("Would you like to save the model (1/0) : ")

    if save :
        model.save('LSTM_ELU.h5')
