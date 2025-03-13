import numpy as np
import pandas as pd

movies = pd.read_excel(r"C:\Users\Scaria\Documents\movienames.xlsx")
credits = pd.read_excel(r"C:\Users\Scaria\Documents\moviecredits.xlsx")
selected_columns = ['genres', 'id', 'keywords', 'overview', 'title', 'cast', 'crew']
movies = movies.merge(credits,on='title')
available_columns = [col for col in selected_columns if col in movies.columns]
print(f"Available columns: {available_columns}")
print(f"Missing columns: {[col for col in selected_columns if col not in movies.columns]}")

# only kaam ke columns , faltu vale droped
movies = movies[['genres','id','keywords','overview','title','cast','crew']]

movies =movies.dropna()

# genres vale column me se sirf genre ka naam fetch
movies['genres'] = movies['genres'].fillna('[]')  # Replace NaN with empty list string

import ast
def genres (obj):
  list =[]
  for i in ast.literal_eval(obj):
    list.append(i['name'])
  
  return list

movies['genres'] =movies['genres'].apply(genres)

# keywords vale column se sirf keyword fetch
def keywords (obj):
  list =[]
  for i in ast.literal_eval(obj):
    list.append(i['name'])
  
  return list

movies['keywords'] =movies['keywords'].apply(genres)

# cast se main 3 cast fetch
def cast (obj):
  list =[]
  j =0
  for i in ast.literal_eval(obj):
    if j != 3:
      list.append(i['name'])
      j +=1
    else:
      break
  
  return list

movies['cast'] =movies['cast'].apply(cast)

# crew se only director name fetch
# def crew (obj):
#   list =[]
#   for i in ast.literal_eval(obj):
#     if i['job'] =='Director':
#       list.append(i['name'])
#       break
  
#   return list

# movies['crew'] =movies['crew'].apply(crew)
import json

def crew(obj):
    try:
        data = json.loads(obj)
        return [i['name'] for i in data if i.get('job') == 'Director']
    except (json.JSONDecodeError, TypeError):
        return []

movies['crew'] = movies['crew'].apply(crew)

# string ko list me convert , taki concate ho sake
movies['overview'] =movies['overview'].apply(lambda x: x.split())

# 'raja ram' -----convert to------> 'rajaram'
movies['cast']= movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']= movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies['genres']= movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']= movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])

# saaf kiya huva dataset , jo ab kaam karne layak hai
movies['tags'] = (movies['cast'] * 9) + (movies['genres'] * 100) + movies['crew'] + movies['keywords'] + movies['overview']

movies =movies.drop(['genres','keywords','overview','cast','crew'],axis=1)

# isi ke basis pe recommendation hoga, converting list into a string
movies['tags'] =movies['tags'].apply(lambda x: " ".join(x))

# words ko vector form me convert, and remove stop_words-------> is, to, and, or
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# vector dekhna ,-----> 
# print(cv.get_feature_names_out())

# (loving, loved, love) ------convert to----------> (love, love, love)
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(text):
  y =[]
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)
movies['tags'] =movies['tags'].apply(stem)

# ek vector se dusre vetor ke bich similarity (similar movie) calculate
from sklearn.metrics.pairwise import cosine_similarity
similarity =cosine_similarity(vectors)

# main movie recommend code
def recommend(movie):
  movie_index = movies[movies['title']== movie].index[0]
  distances = similarity[movie_index]
  movie_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

  for i in movie_list:
    print(movies.iloc[i[0]].title)
recommend('Johnny English Reborn')   # 5 most similar movies recommed karega