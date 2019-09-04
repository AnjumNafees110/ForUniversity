from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Convolution1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout

import mysql
import mysql.connector
import re
import string
from nltk.corpus import stopwords
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# define documents
from itertools import chain
#import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE

#import plotly.express as go
#import plotly.plotly as py

import chart_studio.plotly as py
import plotly.graph_objects as go


cnx = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1',
                              database='new_schema')
print(cnx)
cursor = cnx.cursor()

queryString = "SELECT * FROM new_table3 ORDER BY RAND() LIMIT 0,4000"
cursor.execute(queryString)
fetchrows=4000
rows = cursor.fetchmany(fetchrows)
messages=[]
labels=[]
for row in rows:
    messages.append(str(row[1]))
    labels.append(int(row[2]))
    
#messages=['hi.this is test','welcome bro test','aweomse dance',"who was that?"]
messages1=[]

for i in messages:
    tokens = word_tokenize(i)
    #print(tokens)
    
    
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in tokens]
    tokenized_reports = [word_tokenize(report) for report in stemmed]
    # View tokenized_reports
    #print(tokenized_reports)
        
    regex = re.compile('[%s]' % re.escape(string.punctuation)) #see documentation here: http://docs.python.org/2/library/string.html
    
    tokenized_reports_no_punctuation = []
    
    for review in tokenized_reports:
    
        new_review = []
        for token in review: 
            new_token = regex.sub(u'', token)
            if not new_token == u'':
                new_review.append(new_token)
    
        tokenized_reports_no_punctuation.append(new_review)
    
    #print(tokenized_reports_no_punctuation)
       
    tokenized_reports_no_stopwords = []
    for report in tokenized_reports_no_punctuation:
        new_term_vector = []
        for word in report:
            if not word in stopwords.words('english'):
                new_term_vector.append(word)
        tokenized_reports_no_stopwords.append(new_term_vector)
    
    #print(tokenized_reports_no_stopwords)
    

    tokenized_reports_no_stopwords1 = [x for x in tokenized_reports_no_stopwords if x != []]
    #print(tokenized_reports_no_stopwords1)
    #print(list(chain.from_iterable(tokenized_reports_no_stopwords1)))
    tokenized_reports_no_stopwords2=list(chain.from_iterable(tokenized_reports_no_stopwords1))
    messages1.append(tokenized_reports_no_stopwords2)
    #print(messages1)
    
messages2=[]
for k in messages1:
    #print(k)
    t=" ".join(str(x) for x in k)
    messages2.append(t)
t = Tokenizer()
t.fit_on_texts(messages2)
#print(t.word_counts)
#print(t.document_count)
#print(t.word_index)
#print(t.word_docs)
vocab_size = len(t.word_index) + 1
print(vocab_size)
print("This is vocabulary Size")

# integer encode the documents
encoded_docs = t.texts_to_sequences(messages2)
#print(encoded_docs)

# pad documents to a max length of 300 words
max_length = 300
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
#print(padded_docs)
#print(padded_docs.shape)
 
embeddings_index = dict()
f = open('C:\glove.6b\glove.6B.300d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((vocab_size, 300))
#print(embedding_matrix)
#print(embedding_matrix.shape)
# 
for word, i in t.word_index.items():
#     
     embedding_vector = embeddings_index.get(word)
     if embedding_vector is not None:
         # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
#   
#print(embedding_matrix)

size=int(0.8*fetchrows)
size1=int(0.2*fetchrows)
print(size)
print(size1)

model = Sequential()
e = Embedding(input_dim=vocab_size, output_dim=300, input_length=max_length, weights=[embedding_matrix], trainable=False)
model.add(e)

model.add(Convolution1D(128, 3, padding='same'))
model.add(Convolution1D(64, 3, padding='same'))
model.add(Convolution1D(32, 3, padding='same'))

model.add(Flatten())

model.add(Dense(300,activation='sigmoid'))
model.add(Dropout(0.4))

model.add(Dense(300,activation='sigmoid'))
model.add(Dropout(0.4))

model.add(Dense(300,activation='sigmoid'))
model.add(Dropout(0.4))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

history = model.fit(x=padded_docs[:size], y=labels[:size], validation_split=0.33, batch_size = 15, epochs = 100)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
 
loss, accuracy = model.evaluate(padded_docs[:size], labels[:size], verbose=0)
print('Training Accuracy: %f' % (accuracy*100))
print('Training Loss: %f' % loss)
loss1, accuracy1 = model.evaluate(padded_docs[-size1:], labels[-size1:], verbose=0)
print('Testing Accuracy : %f' % (accuracy1*100))
print('Testing Loss: %f' % loss1)

y_predict = model.predict(padded_docs[-size1:], verbose=0)
yhat_classes = model.predict_classes(padded_docs[-size1:], verbose=0)

# reduce to 1d array
yhat_probs = y_predict[:, 0]
print(yhat_probs)
yhat_classes = yhat_classes[:, 0]
print("yhat_classes")
print(yhat_classes)

f=labels[-size1:]
print("labels [-size1:]")
print(f)

#*****************************************
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
x = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x, acc, 'b', label='Training acc')
plt.plot(x, val_acc, 'r', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(x, loss, 'b', label='Training loss')
plt.plot(x, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()

#************************************

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(labels[-size1:], yhat_classes)
print('Accuracy: %f' % accuracy)

# precision tp / (tp + fp)
precision = precision_score(yhat_classes, labels[-size1:])
print('Precision: %f' % precision)

# recall: tp / (tp + fn)
recall = recall_score(yhat_classes, labels[-size1:])
print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(yhat_classes, labels[-size1:])
print('F1 score: %f' % f1)

# kappa
kappa = cohen_kappa_score(labels[-size1:], yhat_classes)
print('Cohens Kappa: %f' % kappa)

# ROC AUC
auc = roc_auc_score(labels[-size1:], yhat_probs)
print('ROC AUC: %f' % auc)

# confusion matrix
matrix = confusion_matrix(yhat_classes, labels[-size1:])
print(matrix)




