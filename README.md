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
      - [link] (https://storage.googleapis.com/kaggle-data-sets/568973/1032238/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20211209%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20211209T181021Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=1507ddf27d6c77244672752de6ce88c4afe13353b19a170a06df68384d19a0715a1855ae312d9233cea5dee347ee7e50415662457e772d631ab516b545da878382c1472a72bf32a74b32e3871c9f61ecdd8511dc0a4a31bab96df47faa6f99636d5561568dc669cbb8db558018db7c26ead82af0561a6ed84cee975809fdf1391c61208d4787694cc7a48fcbff67cf896d025c77f2d3a356f3a571436903243a6075f5b8a863e51b24dc041a4fc5dbdb6435ea065f20d637d891ec3f28f9665b280fbd4e167c1c9a24d9dd380046eddd2d64129fe72435bf30ee6f5b93cc97d17bde7bf37b90796b9f6a21a9b39193131aca3c06718302061a315cf0883f8425)
       

        * `tracks.csv`: per track metadata such as ID, title, artist, genres, tags and play counts, for all 106,574 tracks.
        * `genres.csv`: all 163 genre IDs with their name and parent (used to infer thegenre hierarchy and top-level genres).
        * `features.csv`: common features extracted with [librosa].
        * `echonest.csv`: audio features provided by [Echonest] (now [Spotify]) for a subset of 13,129 tracks.


Machine Learning :


Neural Network :


# Music Generation

## Data :
