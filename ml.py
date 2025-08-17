#1.Data Collection
import numpy as np
import pandas as pd
fake = pd.read_csv("C:\\CODE\\Python\\Machine Learning\\Fake.csv")
real= pd.read_csv("C:\\CODE\\Python\\Machine Learning\\True.csv")

#2.Data Preprocessing
print(fake.head())
print(real.head())

fake['label'] = 1
real['label'] = 0

#Merging
news = pd.concat([fake, real], axis=0)
print(news.head())

print("*"*50)
print(news.head())
print("*"*50)
print(news.tail())
#Cheking null value
print("*"*50)
print(news.isnull().sum())
print("*"*50)
#Taking entire data and resuffle it
print(news.sample(frac=1).head())
print("*"*50)
# Reset the DataFrame index (old index becomes a new 'index' column)
print(news.reset_index(inplace=True))
print("*"*50)
# Drop the old index column since it's no longer needed
news.drop('index', axis=1, inplace=True)
print("*"*50)
# Preview the first rows to verify the updated index
print(news.head())
print("*"*50)
#3.Feature Extraction
import re
def wordopt(text):
    # handle NaN / non-string values
    if not isinstance(text, str):
        if pd.isna(text):
            return ''
        text = str(text)
    text = text.lower()  # lowercase
    #remove url's
    text = re.sub(r'https?://\S+','',text)
    #remove HTML tags
    text = re.sub(r'<.*?>','',text)
    #remove punctuation
    text = re.sub(r'[^\w\s]','',text)
    #remove digits
    text = re.sub(r'\d','',text)
    #remove newline characters
    text = re.sub(r'\n','',text)

    return text
print("-"*50)
# ensure dtype is string before applying
news['text'] = news['text'].astype(str).apply(wordopt)
print(news['text'])

print("-"*50)
#Train and test data selection

x=news['text']
y=news['label']

print("-"*50)
print(x)
print("-"*50)
print(y)

print("-"*50)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

print(x_train.shape)

print(x_test.shape)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()

#Stored in sparse matrix form of numbers
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Inspect the sparse TF-IDF feature matrix for the training set (rows: docs, cols: vocabulary terms)
print("feature matrix for the training set", xv_train)
# Inspect the sparse TF-IDF feature matrix for the test set (note: ideally use vectorization.transform to avoid data leakage)
print("feature matrix for the test set", xv_test)
print("-"*50)
#Logistic regression(classification)

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train, y_train)

pred_lr = LR.predict(xv_test)

print("Prediction : ")
print(pred_lr)

print(LR.score(xv_test, y_test))

print("Classification Report : ")
from sklearn.metrics import classification_report
print(classification_report(y_test, pred_lr))

#Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
DTC = DecisionTreeClassifier()
DTC.fit(xv_train, y_train)
pred_dtc = DTC.predict(xv_test)
print("Decision Tree Classifier Score: ", DTC.score(xv_test, y_test))

print("Classification Report : ")
print(classification_report(y_test, pred_dtc))


rfc = RandomForestClassifier()
rfc.fit(xv_train, y_train)
pred_rfc = rfc.predict(xv_test)
print(pred_rfc)
print("Random Forest Classifier Score: ", rfc.score(xv_test, y_test))

print("Classification Report : ")
print(classification_report(y_test, pred_rfc))

from sklearn.ensemble import GradientBoostingClassifier
sbc = GradientBoostingClassifier()
sbc.fit(xv_train, y_train)
pred_sbc = sbc.predict(xv_test)
print(pred_sbc)
print("Gradient Boosting Classifier Score: ", sbc.score(xv_test, y_test))

print("Classification Report : ")
print(classification_report(y_test, pred_sbc))


#Output
def output_label(n):
    if n==0:
        return "It is Fake News"
    else:
        return "it is True News"
    

def manual_testing(news):
    testing_news = [{"text": news}]
    testing_df = pd.DataFrame(testing_news)
    testing_df['text'] = testing_df['text'].astype(str).apply(wordopt)
    new_x_test = testing_df['text']
    xv_test_manual = vectorization.transform(new_x_test)
    pred_manual = LR.predict(xv_test_manual)
    pred_gbc = sbc.predict(xv_test_manual)
    pred_rfc = rfc.predict(xv_test_manual)
    return output_label(pred_manual[0])

news_article = str(input("Your news article text here: "))
print(manual_testing(news_article))

manual_testing(news_article)