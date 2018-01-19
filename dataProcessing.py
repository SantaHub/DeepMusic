import pandas as pd

tracks = pd.read_csv('tracks.csv')
features = pd.read_csv('features.csv')
echonest = pd.read_csv('echonest.csv')

#Replace NaN with -1
tracks.fillna(-1)
features.fillna(-1)
echonest.fillna(-1)

#save back to csv. Check code
