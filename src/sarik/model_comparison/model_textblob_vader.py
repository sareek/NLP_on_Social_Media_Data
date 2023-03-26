# import needed libraries
import pandas as pd 

# import graphing library
import matplotlib.pyplot as plt
import os 
import re
import nltk
from nltk.stem import WordNetLemmatizer
#Perceptron classifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay,classification_report 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# display and set working/data directory
os.getcwd()
os.chdir('D:/spring 23/724-DataAnalytics_SocialMedia/Project/github/NLP_on_Social_Media_Data/src/sarik/data_preproccessing')
os.getcwd()

combined_df = pd.read_csv("../../../data/project_data.2023-03-11_13.11.55.783862.csv", sep=",", index_col=0)

duplicated_rows = combined_df.duplicated().sum()
combined_df = combined_df.drop_duplicates()

def clean_tweets(tweets):
    # Stopword removal, converting uppercase into lower case, and lemmatization
    stopwords = nltk.corpus.stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    nltk.download('stopwords')
    data_without_stopwords = []
    for i in range(0, len(tweets)):
        doc = re.sub('[^a-zA-Z]', ' ' , tweets[i])
        doc = doc.lower()
        doc = doc.split()
        doc = [lemmatizer.lemmatize(word) for word in doc if not word in set(stopwords)]
        doc = ' '.join(doc)
        data_without_stopwords.append(doc)
        
    return data_without_stopwords


#get tweets from dataframe
tweets = [i for i in combined_df["text"]]
#add clean_text column with list of clean text
combined_df['clean_text'] = cleaned_tweets

#optional step just reorderning
new_order = [
    'conversation_id', 
    'lang', 
    'reply_settings', 
    'created_at', 
    'clean_text',
    'text',
    'author_id',
    'referenced_tweets',
    'id',
    'edit_history_tweet_ids',
    'public_metrics.retweet_count',
    'public_metrics.reply_count',
    'public_metrics.like_count',
    'public_metrics.impression_count',
    'in_reply_to_user_id',
    'geo.place_id',
    'withheld.copyright',
    'withheld.country_codes',
    'geo.coordinates.type',
    'geo.coordinates.coordinates'
]
combined_df = combined_df.reindex(columns=new_order)

combined_df.head()
#Statistics on the data, mean, median, std deviation, etc.
combined_df.describe()


#########################################TEXTBLOB MODEL#################################

from textblob import TextBlob
#function to get the subjectivity 
def get_subjectivity(tweet):
    return TextBlob(tweet).sentiment.subjectivity

#create a function to get the polarity
def get_polarity(tweet):
    return TextBlob(tweet).sentiment.polarity

#create two new columns
combined_df['Subjectivity'] = combined_df['clean_text'].apply(get_subjectivity)
combined_df['Polarity'] = combined_df['clean_text'].apply(get_polarity)

# categorize as positive, very positive, neutral, negative, and very negative 
def get_stats(score):
    if score < -0.5:
        return 'Very Negative'
    if score >= -0.5 and score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    elif score > 0 and score <= 0.5:
        return 'Positive'
    elif score > 0.5:
        return 'Very Positive'
    
combined_df['Stats'] = combined_df['Polarity'].apply(get_stats)

###EVALUATE CORRECTNESS OF MODEL###
#PERCEPTRON MODEL USED, CAN TRY WITH OTHER CLASSIFIER AS WELL IN FUTURE



def train_test_split_fun(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X_textblob = combined_df[['clean_text']]
y_textblob = combined_df['Stats']
X_train_tb, X_test_tb, y_train_tb, y_test_tb = train_test_split_fun(X_textblob, y_textblob)

model_tb = Perceptron(shuffle=True, random_state=17,max_iter=1000)
vectorizer = TfidfVectorizer()
column_transformer = ColumnTransformer(
    [('tfidf', vectorizer, 'clean_text')])

fitted_model_v = Pipeline([
    ('preprocessing', column_transformer),
    ('model', model_tb)
])

fitted_model_v.fit(X_train_tb, y_train_tb)

y_pred_tb = fitted_model_v.predict(X_test_tb) # trained model makes predictions on 
# y_pred_v

print("Accuracy:",round(accuracy_score(y_test_tb, y_pred_tb)*100,2))
print(classification_report(y_test_tb, y_pred_tb))

cMatrix_tb = confusion_matrix(y_test_tb, y_pred_tb, labels=model_tb.classes_ )

disp_tb = ConfusionMatrixDisplay(confusion_matrix = cMatrix_tb, display_labels=model_tb.classes_)

fig, ax = plt.subplots(figsize=(10, 8))

disp_tb.plot(ax=ax)
plt.show()

# Accuracy: 90.87
#                precision    recall  f1-score   support

#      Negative       0.88      0.83      0.85       934
#       Neutral       0.93      0.97      0.95      3153
#      Positive       0.90      0.89      0.89      2009
# Very Negative       0.71      0.58      0.64        83
# Very Positive       0.83      0.77      0.80       280

#      accuracy                           0.91      6459
#     macro avg       0.85      0.81      0.83      6459
#  weighted avg       0.91      0.91      0.91      6459

##Confusion Matrix picture in visualization directory, name: perceptron_textblob_classifier.png

words = ' '.join( [tweets for tweets in combined_df['clean_text']])
word_cloud = WordCloud(width=800, height=400, random_state=21, max_font_size=40).generate(words)

plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()
#########################################END TEXTBLOB MODEL#################################

#########################################VADER SENTIMENT MODEL#################################
# Install: pip install vaderSentiment

combined_df_m2 = combined_df
combined_df_m2.drop(['Polarity', 'Subjectivity'], axis=1)


vader_model = SentimentIntensityAnalyzer()

def sentiment_analysis(x):
    sentiment_scores = vader_model.polarity_scores(x)
    Polarity = sentiment_scores['compound']
    Subjectivity = 1 - sentiment_scores['neu']
    return Polarity, Subjectivity

combined_df_m2[['Polarity', 'Subjectivity']] = combined_df_m2['clean_text'].apply(lambda x: pd.Series(sentiment_analysis(x)))

def get_stats(score):
    if score < -0.5:
        return 'Very Negative'
    if score >= -0.5 and score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    elif score > 0 and score <= 0.5:
        return 'Positive'
    elif score > 0.5:
        return 'Very Positive'
    
combined_df_m2['Stats'] = combined_df_m2['Polarity'].apply(get_stats)

### MODEL EVALUATION ###

X_vader = combined_df_m2[['clean_text']]
y_vader = combined_df_m2['Stats']

X_train_v, X_test_v, y_train_v, y_test_v = train_test_split_fun(X_vader, y_vader)

model_v = Perceptron(shuffle=True, random_state=17,max_iter=1000)

vectorizer = TfidfVectorizer()
column_transformer = ColumnTransformer(
    [('tfidf', vectorizer, 'clean_text')])

fitted_model_v.fit(X_train_v, y_train_v)

y_pred_v = fitted_model_v.predict(X_test_v) # trained model makes predictions on 

print("Accuracy:",round(accuracy_score(y_test_v, y_pred_v)*100,2))
# Accuracy: 82.8

print(classification_report(y_test_v, y_pred_v))

#                precision    recall  f1-score   support

#      Negative       0.72      0.66      0.69       850
#       Neutral       0.90      0.88      0.89      1607
#      Positive       0.81      0.83      0.82      1795
# Very Negative       0.86      0.87      0.86      1180
# Very Positive       0.82      0.84      0.83      1027

#      accuracy                           0.83      6459
#     macro avg       0.82      0.82      0.82      6459
#  weighted avg       0.83      0.83      0.83      6459

cMatrix_v = confusion_matrix(y_test_v, y_pred_v, labels=model_v.classes_ )

disp_v = ConfusionMatrixDisplay(confusion_matrix = cMatrix_v, display_labels=model_v.classes_)

fig, ax = plt.subplots(figsize=(10, 8))

disp_v.plot(ax=ax)
plt.show()

#see visualization/images name: perceptron_vader_classifier.png