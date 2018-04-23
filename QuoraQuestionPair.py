
# coding: utf-8

# In[1]:


import os,sys
import pandas as pd
import numpy as np


# In[2]:


os.chdir('E:\\Kaggle\\QuoraQuestionPair')


# In[5]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras.models import Sequential
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers import Dense ,Flatten


# In[3]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[8]:


embed_size = 50 # how big is each word vector
max_features = 30000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use


# In[9]:


train.headd()


# In[13]:


y = train["is_duplicate"].values

list_sentences_train1 = train["question1"].fillna("_na_").values
list_sentences_test1 = test["question1"].fillna("_na_").values
tokenizer1 = Tokenizer(num_words=max_features)
tokenizer1.fit_on_texts(list(list_sentences_train1))
list_tokenized_train1 = tokenizer1.texts_to_sequences(list_sentences_train1)
list_tokenized_test1 = tokenizer1.texts_to_sequences(list_sentences_test1)
X_t1 = pad_sequences(list_tokenized_train1, maxlen=maxlen)
X_te1 = pad_sequences(list_tokenized_test1, maxlen=maxlen)

list_sentences_train2 = train["question2"].fillna("_na_").values
list_sentences_test2 = test["question2"].fillna("_na_").values
tokenizer2 = Tokenizer(num_words=max_features)
tokenizer2.fit_on_texts(list(list_sentences_train2))
list_tokenized_train2 = tokenizer2.texts_to_sequences(list_sentences_train2)
list_tokenized_test2 = tokenizer2.texts_to_sequences(list_sentences_test2)
X_t2 = pad_sequences(list_tokenized_train2, maxlen=maxlen)
X_te2 = pad_sequences(list_tokenized_test2, maxlen=maxlen)


# In[15]:


X_t2.shape


# In[19]:


X_train = np.concatenate((X_t1,X_t2),axis = 1)


# In[21]:


def get_coefs(word,*arr):
    return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open('E:\\Kaggle\\Toxic_Comment\\glove.6B.50d.txt\\glove.6B.50d.txt', encoding="utf8"))


# In[37]:


all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std
w1 = tokenizer1.word_index
w2 = tokenizer2.word_index
w1.update(w2)

word_index = w1
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[44]:


inp = Input(shape=(200,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(20, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model.fit(X_train, y, batch_size=32, epochs=2,validation_split= 0.1) # validation_split=0.1);
#Train on 363861 samples, validate on 40429 samples
#Epoch 1/2
#  1664/363861 [..............................] - ETA: 2624s - loss: 0.6617 - acc: 0.6220


# In[ ]:


#### trying CNN


# In[50]:



model = Sequential()
model.add(Embedding(max_features, embed_size, weights=[embedding_matrix], input_length=200))
model.add(Conv1D(128, 3 ))
model.add(Conv1D(64, 3 ))
model.add(Conv1D(32, 3))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(180,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model.fit(X_train, y, batch_size=32, epochs=2,validation_split= 0.1)
##Train on 363861 samples, validate on 40429 samples
#Epoch 1/2
 # 4288/363861 [..............................] - ETA: 1084s - loss: 9.9528 - acc: 0.3757


# # trying conv2d

# In[48]:


X_train_stack = np.concatenate((X_t1,X_t2))


# In[52]:


X_train_stack.shape


# In[59]:


model = Sequential()
model.add(Embedding(max_features, embed_size, weights=[embedding_matrix], input_length=200))
model.add(Conv2D(10, 3, 3, border_mode='same', input_shape=(100, 100, 1))
##model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=(200,100,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


# In[47]:


def model_conv1D_(emb_matrix):
    
    # The embedding layer containing the word vectors
    emb_layer = Embedding(
        input_dim=emb_matrix.shape[0],
        output_dim=emb_matrix.shape[1],
        weights=[emb_matrix],
        input_length=60,
        trainable=False
    )
    
    # 1D convolutions that can iterate over the word vectors
    conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
    conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
    conv4 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
    conv5 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
    conv6 = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')

    # Define inputs
    seq1 = Input(shape=(60,))
    seq2 = Input(shape=(60,))

    # Run inputs through embedding
    emb1 = emb_layer(seq1)
    emb2 = emb_layer(seq2)

    # Run through CONV + GAP layers
    conv1a = conv1(emb1)
    glob1a = GlobalAveragePooling1D()(conv1a)
    conv1b = conv1(emb2)
    glob1b = GlobalAveragePooling1D()(conv1b)

    conv2a = conv2(emb1)
    glob2a = GlobalAveragePooling1D()(conv2a)
    conv2b = conv2(emb2)
    glob2b = GlobalAveragePooling1D()(conv2b)

    conv3a = conv3(emb1)
    glob3a = GlobalAveragePooling1D()(conv3a)
    conv3b = conv3(emb2)
    glob3b = GlobalAveragePooling1D()(conv3b)

    conv4a = conv4(emb1)
    glob4a = GlobalAveragePooling1D()(conv4a)
    conv4b = conv4(emb2)
    glob4b = GlobalAveragePooling1D()(conv4b)

    conv5a = conv5(emb1)
    glob5a = GlobalAveragePooling1D()(conv5a)
    conv5b = conv5(emb2)
    glob5b = GlobalAveragePooling1D()(conv5b)

    conv6a = conv6(emb1)
    glob6a = GlobalAveragePooling1D()(conv6a)
    conv6b = conv6(emb2)
    glob6b = GlobalAveragePooling1D()(conv6b)

    mergea = concatenate([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a])
    mergeb = concatenate([glob1b, glob2b, glob3b, glob4b, glob5b, glob6b])

    # We take the explicit absolute difference between the two sentences
    # Furthermore we take the multiply different entries to get a different measure of equalness
    diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(4 * 128 + 2*32,))([mergea, mergeb])
    mul = Lambda(lambda x: x[0] * x[1], output_shape=(4 * 128 + 2*32,))([mergea, mergeb])

    # Add the magic features
    magic_input = Input(shape=(5,))
    magic_dense = BatchNormalization()(magic_input)
    magic_dense = Dense(64, activation='relu')(magic_dense)

    # Add the distance features (these are now TFIDF (character and word), Fuzzy matching, 
    # nb char 1 and 2, word mover distance and skew/kurtosis of the sentence vector)
    distance_input = Input(shape=(20,))
    distance_dense = BatchNormalization()(distance_input)
    distance_dense = Dense(128, activation='relu')(distance_dense)

    # Merge the Magic and distance features with the difference layer
    merge = concatenate([diff, mul, magic_dense, distance_dense])

    # The MLP that determines the outcome
    x = Dropout(0.2)(merge)
    x = BatchNormalization()(x)
    x = Dense(300, activation='relu')(x)

    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    pred = Dense(1, activation='sigmoid')(x)

    # model = Model(inputs=[seq1, seq2, magic_input, distance_input], outputs=pred)
    model = Model(inputs=[seq1, seq2, magic_input, distance_input], outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model

model2 = model_conv1D_(embedding_matrix)

