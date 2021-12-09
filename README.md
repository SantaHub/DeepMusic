# DeepMusic
Generation and classification of music using Capsule Neural Networks. (Modified CNN)

## Folder structure 

 -  Presentation
       - test : test
       - Images : contianers images of ML and DL : accuracy, Loss and model code.
       - content.md : Plan for this research paper
       - demo.py : Used to demo the models build in the repo. Loads and predicts music.
- baseline.ipynb : A notebook to find baseline efficiency using sci-kit models
- creation.py : donwloads traims and create zips of music data with normalization.
- featureGen.ipynb : generate and prints features from an MP3.
- feature.py : create features and save them as train and test.
- ml.py : Uses the given classifier and train and test the model.
- Resource : Informations and links to the topic resources
- setup.py : for the ownership of the proejct from baseline.ipynb
- train_network_work.py : Data modeling and triaining for LSTM, CNN.
- utils.py : Downloads and loads track from FMA

# Music Genre Classification 
## Data :
 -  FMA
       - 1L music files. : https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
       - music features generated : librosa : https://github.com/librosa/librosa
- GTZAN Music Genre classification 
      - https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification/download
       

        * `tracks.csv`: per track metadata such as ID, title, artist, genres, tags and play counts, for all 106,574 tracks.
        * `genres.csv`: all 163 genre IDs with their name and parent (used to infer thegenre hierarchy and top-level genres).
        * `features.csv`: common features extracted with [librosa].
        * `echonest.csv`: audio features provided by [Echonest] (now [Spotify]) for a subset of 13,129 tracks.


Machine Learning :


Neural Network :


# Music Generation

## Data :
