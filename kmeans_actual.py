import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

f = open("test.csv", "r") 
x=[]
y=[]
for line in f:
    temp=line.split(',')
    x.append(float(temp[0]))
    y.append(float(temp[1].rstrip('\n')))

print "x from file :",x
print "y from file :",y

df = pd.DataFrame({
    'x': x,
    'y': y
})


k=int(raw_input("Enter number of cluster:"))
np.random.seed(200)

tempx=x
tempy=y    
centroids={}

for i in range(k):
    centroids[i+1]=[tempx.pop(0),tempy.pop(0)]

print "Initial Centroids : " , centroids

def assignment(df, centroids):
    for i in centroids.keys():
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    return df

def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    print "centroids:",centroids    
    return k

df = assignment(df, centroids)
while True:
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df, centroids)
    if closest_centroids.equals(df['closest']):
        print "Matched:" , centroids    
        break