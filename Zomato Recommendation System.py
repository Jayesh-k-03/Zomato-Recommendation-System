#!/usr/bin/env python
# coding: utf-8

# # Zomato Recommendation System

# A recommendation system is a type of information filtering system that predicts or suggests items or actions a user might be interested in based on their preferences, behavior, or past interactions. These systems are commonly used in various online platforms to help users discover relevant products, services, content, or people.

# Recommendation systems are widely used in e-commerce platforms, streaming services, social networks, and other online applications to enhance user experience, increase engagement, and drive revenue by suggesting relevant content or products to users.

# # Types of  Recommendation System

# 1.Content-Based Filtering: This method recommends items similar to those the user has liked or interacted with in the past. It analyzes the attributes or features of items and recommends items with similar characteristics.
# 
# 2.Collaborative Filtering: Collaborative filtering recommends items based on the preferences of other users. It identifies users with similar preferences or behaviors and suggests items that these similar users have liked or interacted with.
# 
# 3.Hybrid Recommendation Systems: These systems combine multiple recommendation techniques to provide more accurate and diverse recommendations. For example, a hybrid system might combine content-based filtering with collaborative filtering to take advantage of the strengths of both approaches.
# 
# 4.Knowledge-Based Recommendation Systems: Knowledge-based recommendation systems use explicit knowledge about users' preferences and domain knowledge about items to generate recommendations. They often involve rules or constraints based on domain expertise.
# 
# 5.Context-Aware Recommendation Systems: Context-aware recommendation systems consider additional contextual information, such as time, location, or device, to provide more personalized recommendations that are relevant to the user's current situation.

# # In this Notebook we are using Content-Based Filtering to suggests similar restaurants based on cosine similarity scores.
#  
#  Some key  point for this project which will help you to understand futher steps 
#  
# 1.Cosine Similarity: The function calculates cosine similarity scores between restaurants based on certain features (likely cuisine, mean rating, and cost). Cosine similarity is a technique commonly used in content-based recommendation systems to measure the similarity between items based on their feature vectors.
#     
# 2.Feature-based Recommendations: The function identifies restaurants with similar feature vectors to the input restaurant (specified by the name parameter) and recommends them based on their similarity scores. This is a characteristic of content-based recommendation systems, which recommend items similar to those that a user has liked or interacted with in the past, based on the attributes or features of the items.
# 
# 3.No User Data Considered: The function doesn't explicitly consider user preferences or behaviors. Instead, it relies solely on the features of the restaurants to make recommendations. This aligns with the content-based filtering approach, which doesn't require information about other users' preferences or behaviors.

# # Import neccessary Libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import re
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


#reading the dataset and loading datasets 
zomato_real=pd.read_csv("D:\Projects\Zomato\zomato.csv")
zomato_real.head() 


# In[3]:


zomato_real.info()


# In[4]:


zomato_real.describe()


# Data Cleaning

# In[5]:


#Deleting Unnnecessary Columns
zomato=zomato_real.drop(['url','dish_liked','phone'],axis=1)
#Dropping the column "dish_liked", "phone", "url" and saving the new dataset as "zomato"


# In[6]:


zomato.duplicated().sum()
zomato.drop_duplicates(inplace=True)


# In[7]:


zomato.isnull().sum()
zomato.dropna(how='any',inplace=True)
zomato.info()


# In[8]:


zomato.columns


# In[9]:


#Changing the column names
zomato = zomato.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type',
                                  'listed_in(city)':'city','rate':'rating'})
zomato.columns


# In[10]:


#Some Transformations
zomato['cost'] = zomato['cost'].astype(str) #Changing the cost to string
zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',','.')) #Using lambda function to replace ',' from cost
zomato['cost'] = zomato['cost'].astype(float) # Changing the cost to Float
zomato.info()


# In[11]:


#Reading Rate of dataset
zomato['rating'].unique()


# In[12]:


# Removing '/5' from Ratings
zomato = zomato.loc[zomato.rating != 'NEW']
zomato = zomato.loc[zomato.rating != '-'].reset_index(drop=True)

# Convert all values in 'rating' column to strings
zomato['rating'] = zomato['rating'].astype(str)

# Remove '/5' from ratings
zomato['rating'] = zomato['rating'].str.replace('/5', '')

# Strip any leading or trailing whitespace
zomato['rating'] = zomato['rating'].str.strip()

# Convert cleaned ratings to float
zomato['rating'] = zomato['rating'].astype(float)

zomato['rating'].head()


# In[13]:


# Adjust the column names
zomato.name = zomato.name.apply(lambda x:x.title())
zomato.online_order.replace(('Yes','No'),(True, False),inplace=True)
zomato.book_table.replace(('Yes','No'),(True, False),inplace=True)
zomato.cost.unique()


# In[14]:


zomato.head(10)


# In[15]:


zomato['city'].unique()


# In[16]:


zomato.head(10)


# In[17]:


## Checking Null values
zomato.isnull().sum()


# In[18]:


## Computing Mean Rating
restaurants = list(zomato['name'].unique())
zomato['Mean Rating'] = 0

for i in range(len(restaurants)):
    zomato['Mean Rating'][zomato['name'] == restaurants[i]] = zomato['rating'][zomato['name'] == restaurants[i]].mean()


# In[19]:


zomato.head(10)


# In[20]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (1,5))

zomato[['Mean Rating']] = scaler.fit_transform(zomato[['Mean Rating']]).round(2)

zomato.sample(10)


# In[21]:


# Calculate the average rating
average_rating = zomato['Mean Rating'].mean()

# Print the average rating
print("Average Rating:", average_rating)


# # Text Preprocessing

# Some of the common text preprocessing / cleaning steps are:
# 
# 1.Lower casing
# 
# 2.Removal of Punctuations
# 
# 3.Removal of Stopwords
# 
# 4.Removal of URLs
# 
# 5.Spelling correction

# In[22]:


zomato[['reviews_list', 'cuisines']].sample(5)


# In[23]:


zomato.head(10)


# In[24]:


## Lower Casing
zomato["reviews_list"] = zomato["reviews_list"].str.lower()
zomato[['reviews_list', 'cuisines']].sample(5)


# In[25]:


# Removal of Puctuations
import string
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_punctuation(text))
zomato[['reviews_list', 'cuisines']].sample(10)


# In[26]:


## Removal of Stopwords
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_stopwords(text))


# In[27]:


## Removal of URLS
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_urls(text))


# In[28]:


zomato[['reviews_list', 'cuisines']].sample(10)


# In[29]:


#RESTAURANT NAMES:
restaurant_names = list(zomato['name'].unique())
restaurant_names


# In[30]:


def get_top_words(column, top_nu_of_words, nu_of_word):
    
    vec = CountVectorizer(ngram_range= nu_of_word, stop_words='english')
    
    bag_of_words = vec.fit_transform(column)
    
    sum_words = bag_of_words.sum(axis=0)
    
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    return words_freq[:top_nu_of_words]
zomato.head()


# In[31]:


zomato.sample(10)


# In[32]:


zomato.shape


# In[33]:


zomato.columns


# In[34]:


import pandas

# Randomly sample 60% of your dataframe
df_percent = zomato.sample(frac=0.5)


# In[35]:


df_percent.shape


# # Term Frequency-Inverse Document Frequency¶
# Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each document. This will give you a matrix where each column represents a word in the overview vocabulary (all the words that appear in at least one document) and each column represents a restaurant, as before.
# 
# TF-IDF is the statistical method of evaluating the significance of a word in a given document.
# 
# TF — Term frequency(tf) refers to how many times a given term appears in a document.
# 
# IDF — Inverse document frequency(idf) measures the weight of the word in the document, i.e if the word is common or rare in the entire document. The TF-IDF intuition follows that the terms that appear frequently in a document are less important than terms that rarely appear. Fortunately, scikit-learn gives you a built-in TfIdfVectorizer class that produces the TF-IDF matrix quite easily.

# In[36]:


df_percent.set_index('name', inplace=True)


# In[37]:


indices = pd.Series(df_percent.index)


# In[38]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Adjust parameters to reduce the size of the TF-IDF matrix
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=5, max_features=5000, stop_words='english')

# Fit and transform the data
tfidf_matrix = tfidf.fit_transform(df_percent['reviews_list'])


# In[40]:


from sklearn.metrics.pairwise import cosine_similarity

# Assuming tfidf_matrix_reduced is already computed
cosine_similarities = cosine_similarity(tfidf_matrix_reduced, dense_output=False)


# In[41]:


from sklearn.decomposition import TruncatedSVD

# Choose the number of components for TruncatedSVD
n_components = 100

# Initialize TruncatedSVD
svd = TruncatedSVD(n_components=n_components)

# Fit TruncatedSVD to the TF-IDF matrix and transform the matrix
tfidf_matrix_reduced = svd.fit_transform(tfidf_matrix)

# Compute cosine similarities using the reduced TF-IDF matrix
cosine_similarities = cosine_similarity(tfidf_matrix_reduced, dense_output=False)


# In[42]:


def recommend(name, cosine_similarities = cosine_similarities):
    
    # Create a list to put top restaurants
    recommend_restaurant = []
    
    # Find the index of the hotel entered
    idx = indices[indices == name].index[0]
    
    # Find the restaurants with a similar cosine-sim value and order them from bigges number
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    
    # Extract top 30 restaurant indexes with a similar cosine-sim value
    top30_indexes = list(score_series.iloc[0:31].index)
    
    # Names of the top 30 restaurants
    for each in top30_indexes:
        recommend_restaurant.append(list(df_percent.index)[each])
    
    # Creating the new data set to show similar restaurants
    df_new = pd.DataFrame(columns=['cuisines', 'Mean Rating', 'cost'])
    
    # Create the top 30 similar restaurants with some of their columns
    for each in recommend_restaurant:
        df_new = df_new.append(pd.DataFrame(df_percent[['cuisines','Mean Rating', 'cost']][df_percent.index == each].sample()))
    
    # Drop the same named restaurants and sort only the top 10 by the highest rating
    df_new = df_new.drop_duplicates(subset=['cuisines','Mean Rating', 'cost'], keep=False)
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)
    
    print('TOP %s RESTAURANTS LIKE %s WITH SIMILAR REVIEWS: ' % (str(len(df_new)), name))
    
    return df_new


# In[43]:


# HERE IS A RANDOM RESTAURANT. LET'S SEE THE DETAILS ABOUT THIS RESTAURANT:
df_percent[df_percent.index == 'Pai Vihar'].head()


# In[44]:


recommend('Pai Vihar')


# In[46]:


# Group data by restaurant type and calculate average rating for each type
average_ratings_by_type = zomato.groupby('rest_type')['Mean Rating'].mean()

# Sort the average ratings in descending order
average_ratings_by_type_sorted = average_ratings_by_type.sort_values(ascending=False)

# Print the restaurant types with the highest average ratings
print("Restaurant types with higher average ratings:")
print(average_ratings_by_type_sorted)


# In[ ]:




