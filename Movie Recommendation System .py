#!/usr/bin/env python
# coding: utf-8

# In[73]:


import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.metrics.pairwise import cosine_similarity


# In[115]:


df=pd.read_csv(r"C:\Users\Akash\Downloads\archive (17)\mymoviedb.csv",lineterminator='\n')
df


# In[120]:


df=df.reset_index()
df


# In[121]:


df.isnull().sum()


# In[122]:


df.head()


# In[123]:


df.shape


# In[124]:


df['Original_Language'].value_counts()


# In[128]:


df.tail()


# In[129]:


#selection of features 
selected_features=['Genre','Title','Overview','Release_Date']
print(selected_features)


# In[130]:


#Replacing the null vlaues with null string


# In[131]:


for feature in selected_features:
    df[feature]=df[feature].fillna('')


# In[132]:


#combining the all selected fearures


# In[133]:


combined_feature=df['Genre']+' '+df['Title']+' '+df['Overview']+' '+df['Release_Date']
print(combined_feature)


# In[134]:


Vectorizer=TfidfVectorizer()
feature_Vectors= Vectorizer.fit_transform(combined_feature)
print(feature_Vectors)


# In[135]:


#getting the similarity score using cosine_similarity


# In[150]:


similarity=cosine_similarity(feature_Vectors)
similarity


# In[137]:


similarity.shape#(index,similarity_score)


# In[138]:


#getting the movie name from the user


# In[139]:


movie_name=input('Enter your fav movie name : ')


# In[140]:


#create a list of all movie names using in the given dataset


# In[141]:


list_of_all_titles=df['Title'].tolist()
print(list_of_all_titles)


# In[142]:


#finding the close match for the movie name given by the user


# In[143]:


find_close_match=difflib.get_close_matches(movie_name,list_of_all_titles)
print(find_close_match)


# In[144]:


close_match=find_close_match[0]
print(close_match)


# In[145]:


#finding the index of the movie with title


# In[147]:


Index_of_the_movie=df[df.Title == close_match]['index'].values[0]
print(Index_of_the_movie)


# In[148]:


#getting a list of similar movies


# In[155]:


similarity_score=list(enumerate(similarity[Index_of_the_movie]))  #index,simillarity_score of avengers movie compare with other movie name given in dataset
similarity_score


# In[156]:


len(similarity_score)


# In[158]:


#only those values which are highest similarity_score
#sortingg the movies based on their similarity_score


# In[159]:


sorted_movie_list=sorted(similarity_score,key=lambda x:x[1],reverse=True)
sorted_movie_list


# In[160]:


#print the name of similar movies


# In[165]:


print('Movie suggested for you : \n')

i=1

for movie in sorted_movie_list:
    index=movie[0]
    title_from_index=df[df.index==index]['Title'].values[0]
    if(i<32):
        print(i,'.',title_from_index)
        i=i+1


# Movie Recommendation system

# In[167]:


movie_name=input('Enter your fav movie name : ')

list_of_all_titles=df['Title'].tolist()

find_close_match=difflib.get_close_matches(movie_name,list_of_all_titles)

close_match=find_close_match[0]

Index_of_the_movie=df[df.Title == close_match]['index'].values[0]

similarity_score=list(enumerate(similarity[Index_of_the_movie]))

sorted_movie_list=sorted(similarity_score,key=lambda x:x[1],reverse=True)

print('Movie suggested for you : \n')

i=1

for movie in sorted_movie_list:
    index=movie[0]
    title_from_index=df[df.index==index]['Title'].values[0]
    if(i<32):
        print(i,'.',title_from_index)
        i=i+1


# In[ ]:




