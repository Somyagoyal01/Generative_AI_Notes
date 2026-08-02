import streamlit as st #can use flask also instead of streamlit
import nltk
import sklearn
import pandas as pd
import pickle
import joblib
#we use joblib because we need to push the file into github for deployment through render,azure,aws etc and github has a memory limit of 100mb, for pickle this limit exceeds but in joblib there is a compress function which can compress the file from 100 to 50 mb.
st.title('Movie Recommendation System')
with open('movies.pickle','rb') as m:
    movies=pickle.load(m)
similarity=joblib.load('similarity.joblib')
movie_names=movies['title'].values
def recommend(name_movie):
    movie_index=movies[movies['title']==name_movie].index[0]
    recommendations=similarity[movie_index]
    movie_list=sorted(enumerate(recommendations),reverse=True,key=lambda x:x[1])[1:6]
    recommended_movies=[]
    for i in movie_list:
       recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies
    
name_movie=st.selectbox('Enter the movie name',movie_names)
if st.button('Recommend'):
    r=recommend(name_movie)
    st.write('The recommended movies are:')
    for i in r:
        st.write(i)
