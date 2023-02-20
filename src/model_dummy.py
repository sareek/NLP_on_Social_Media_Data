#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import needed libraries
import pandas as pd 

# import graphing library
import matplotlib.pyplot as plt

import os  # for OS interface (to get/change directory)
# display and set working/data directory
os.getcwd()
os.chdir('D:/spring 23/724-DataAnalytics_SocialMedia/hate_speech')
os.getcwd()

# import the data; note the field separator
olid_data_orig = pd.read_csv("olid-training-v1.0.tsv", sep="\t")
df = pd.read_csv("olid-training-v1.0.tsv", sep="\t")


# In[4]:


df.head()
df[df.columns[0]].count()


# In[5]:


import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")


# In[6]:


# Define a function to extract linguistic features from a tweet
def get_linguistic_features(tweet):
    doc = nlp(tweet)
    features = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.like_num:
            features.append(token.lemma_.lower())
    return features


# In[8]:


from collections import Counter
# Count the frequency of linguistic features in each class
linguistic_feature_counts = {
    "OFF": Counter(),
    "NOT": Counter()
}


# In[9]:


# # Loop through each tweet in the data
# for i in range(len(df)):
#     tweet = df.loc[i, 'tweet']
#     label = df.loc[i, 'subtask_a']
#     doc = nlp(tweet)
    
#     # Extract linguistic features
#     features = set()
#     for token in doc:
#         if token.is_stop:
#             features.add(token.lemma_)
#         if token.pos_ in ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB']:
#             features.add(token.lemma_)
#         if token.pos_ == 'VERB' and token.dep_ in ['aux', 'auxpass']:
#             features.add(token.lemma_)
#         if token.ent_type_ in ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']:
#             features.add(token.ent_type_)
    


# In[10]:


for _, row in df.iterrows():
    label = row["subtask_a"]
    tweet = row["tweet"]
    features = get_linguistic_features(tweet)
    linguistic_feature_counts[label].update(features)


# In[13]:


features


# In[11]:


# Print the top 10 most frequent linguistic features for each class
print("Top 10 most frequent linguistic features in offensive tweets:")
print(linguistic_feature_counts["OFF"].most_common(20))


# In[12]:


print("\nTop 10 most frequent linguistic features in non-offensive tweets:")
print(linguistic_feature_counts["NOT"].most_common(20))


# In[ ]:




