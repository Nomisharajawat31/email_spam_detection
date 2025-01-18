#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv("spamdata.csv")


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.shape


# In[8]:


df.describe()


# In[9]:


df.isnull().sum()


# In[10]:


df.duplicated().sum()


# In[11]:


pd.crosstab(df['Unnamed: 2'],df['Unnamed: 4'],margins=False)


# # 1- DATA Cleaning

# In[12]:


#drop last 3 columns
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[13]:


#Renaming the colums
df.rename(columns={'v1':'target','v2':'text'},inplace=True)


# In[14]:


from sklearn.preprocessing import LabelEncoder
encoder= LabelEncoder()


# In[15]:


df['target']=encoder.fit_transform(df['target'])


# In[16]:


df.head()


# In[17]:


#Remove duplicate values
df=df.drop_duplicates(keep='first')


# In[18]:


df.duplicated().sum()


# # 2-EDA=Exporatary data analyst

# In[19]:


df['target'].value_counts()


# In[20]:


import matplotlib.pyplot as plt


# In[21]:


plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[22]:


get_ipython().system('pip install nltk')


# In[23]:


import nltk


# In[24]:


nltk.download('punkt')


# In[25]:


df['num_char']=df['text'].apply(len)


# In[26]:


df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[27]:


df.head()


# In[28]:


df['num_sent']=df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))


# In[29]:


df[['num_char','num_words','num_sent']].describe()


# In[30]:


#ham messages
df[df['target']==0][['num_char','num_words','num_sent']].describe()


# In[31]:


#spam messages
df[df['target']==1][['num_char','num_words','num_sent']].describe()


# In[32]:


import seaborn as sns


# In[33]:


(df[df['target']==0]['num_char']).hist(bins=80),(df[df['target']==1]['num_char']).hist(bins=20)
plt.figure(figsize=(12,6))
#(df[df['target']==1]['num_char']).hist(bins=20)


# In[34]:


(df[df['target']==0]['num_words']).hist(bins=80),(df[df['target']==1]['num_words']).hist(bins=20)
plt.figure(figsize=(12,6))


# In[35]:


(df[df['target']==0]['num_sent']).hist(bins=80),(df[df['target']==1]['num_sent']).hist(bins=20)
plt.figure(figsize=(12,6))


# In[36]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_char'])
sns.histplot(df[df['target']==1]['num_char'],color='red')


# In[37]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_sent'])
sns.histplot(df[df['target']==1]['num_sent'],color='red')


# In[38]:


sns.pairplot(df,hue='target')


# In[39]:


sns.heatmap(df.corr(),annot=True)


# # 3- Data Preprocessing
#>lower case
#>tokenization= broken into words
#>removing special characters
#>stemming=  convert similar words eg:dance,dancer,dancing into dance 
# In[40]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords.words('english')
import string


# In[41]:


def transform_text(text):
    text= text.lower()
    
    text= nltk.word_tokenize(text)
#removing special char    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
            
            
#removing stopwords and puntuation            
    text=y[:]
    y.clear()
    
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
   


    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y) 


# In[43]:


transform_text('Is that seriously how you spell his name?')


# In[44]:


from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()
ps.stem('dancers')


# In[45]:


df['text'][20]


# In[46]:


df['transformed_text']=df['text'].apply(transform_text)


# In[47]:


df.head()


# In[48]:


get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud
wc=WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[49]:


spam_wc= wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=' '))


# In[50]:


plt.figure(figsize=(12,10))
plt.imshow(spam_wc)


# In[51]:


ham_wc= wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" "))


# In[52]:


plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[53]:


spam_corpus = []
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[54]:


len(spam_corpus)


# In[55]:


get_ipython().system('pip install collections')
from collections import Counter
counter=Counter(spam_corpus)
most_common= counter.most_common(30)
#df=pd.DataFrame(most_common,columns=['word','count'])
#sns.barplot(x='word',y='count',data=df)
plt.xticks(rotation='vertical')
plt.show()
#sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])


# In[56]:


ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[57]:


len(ham_corpus)


# In[58]:


print(df.columns)


# In[59]:


print(df.head())


# # 4- Model Building

# In[94]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv= CountVectorizer()
tfidf=TfidfVectorizer(max_features=3000)


# In[95]:


x= tfidf.fit_transform(df['transformed_text']).toarray()


# In[106]:


#from sklearn.preprocessing import MinMaxScaler
#scaler=MinMaxScaler()
#x=scaler.fit_transform(x)


# In[107]:


y=df['target'].values


# In[108]:


y


# In[109]:


from sklearn.model_selection import train_test_split


# In[110]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[111]:


from sklearn.naive_bayes import GaussianNB , MultinomialNB , BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[112]:


gnb=GaussianNB()
mnb= MultinomialNB()
bnb= BernoulliNB()


# In[113]:


gnb.fit(x_train,y_train)
y_pred1=gnb.predict(x_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[114]:


mnb.fit(x_train,y_train)
y_pred2=mnb.predict(x_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[115]:


bnb.fit(x_train,y_train)
y_pred3=bnb.predict(x_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[116]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




