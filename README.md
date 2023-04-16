# MOVIE RECOMMENDATION SYSTEM DATA WARE HOUSE 


### ALGORITHM :


STEP 1:  Import the necessary libraries.

STEP 2:  Load the related dataset to the given project.

STEP 3:  Load the movie dataset from kaggle and find the information such as shape and info from the dataset.

STEP 4:  Replace the null values with null string in the given dataset.

STEP 5:  Combining all the 5 selected features.

STEP 6:  Getting the similarity scores using cosine similarity in the given dataset.

STEP 7:  Getting the movie name from the user and creating a list with all the movie names given in the dataset.

STEP 8:  Finding the close match for the movie name given by the user.

STEP 9:  Finding the index of the movie with title.

STEP 10:  Getting a list of similar movies and sorting the movies based on their similarity score

STEP 11:  Print the name of similar movies based on the index.

STEP 12:  Finally plot the graph to analyse the popularity in the given dataset movies.



## SOURCE CODE : 
```python3
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
data = pd.read_csv("movies.csv")
data.head()
columns =  ['genres','keywords','tagline','cast','director']
print(columns)
for feature in columns:
    data[feature] = data[feature].fillna('')
feature = data['genres']+' '+ data['keywords']+' '+data['tagline']+' '+data['cast']+' '+data['director']
vector = TfidfVectorizer()
vector_feature = vector.fit_transform(feature)
print(vector_feature)
scores = cosine_similarity(vector_feature)
print(scores)
print(scores.shape)
movie_name = input(' Enter Movie Name : ')
titles = data['title'].tolist()
print(titles)
match = difflib.get_close_matches(movie_name, titles)
print(match)
similar_match = match[0]
print(similar_match)
movie = data[data.title == similar_match]['index'].values[0]
print(movie)
similar_movies = list(enumerate(scores[movie]))
print(similar_movies)
sorted_similar_movies = sorted(similar_movies, key = lambda x:x[1], reverse = True) 
print(sorted_similar_movies)
print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
    index = movie[0]
    title = data[data.index==index]['title'].values[0]
    if (i<30):
        print(i, '.',title)
        i+=1
pop= data.sort_values('popularity', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(6),pop['popularity'].head(6), align='center',color='blue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies" )
```



OUTPUT : 


![image](https://user-images.githubusercontent.com/81132849/232331510-ec416e31-10b0-4ccd-a0f8-a80f73ff4efa.png)

![image](https://user-images.githubusercontent.com/81132849/232331534-efccde60-e35b-4ba7-b2d7-3782e9537e04.png)

![image](https://user-images.githubusercontent.com/81132849/232331560-3dfc26ba-9c7d-4f2f-b69d-ae3eef9624ad.png)

![image](https://user-images.githubusercontent.com/81132849/232331613-0e7a90c5-fbd9-42e5-ae62-4801e7193a10.png)



