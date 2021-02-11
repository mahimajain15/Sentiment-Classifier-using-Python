#!/usr/bin/env python
# coding: utf-8

# # Sentiment Classifier using Python
# 
# 1. preprocess the data
# 2. convert English data to numerical representations
# 3. prepare it to be fed as input for our deeplearning model with GRUs.

# ## Importing the modules :

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(42)


# ## Loading the dataset :

# In[2]:


#importing dataset
import tensorflow_datasets as tfds
datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
print(datasets.keys())


# In[3]:


train_size = info.splits["train"].num_examples
test_size = info.splits["test"].num_examples
print(train_size, test_size)


# ## Exploring the dataset :

# In[4]:


for X_batch, y_batch in datasets["train"].batch(2).take(2):
    for review, label in zip(X_batch.numpy(), y_batch.numpy()):
        print("Review : ", review.decode("utf-8")[:200], "...")
        print("Label : ", label, " = Positive" if label else " = Negative")
        print()


# ## Defining the preprocess function :

# In[5]:


def preprocess(X_batch, y_batch):
    X_batch = tf.strings.substr(X_batch, 0, 300)
    X_batch = tf.strings.regex_replace(X_batch, rb"<br\s*/?>", b" ")
    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z]", b" ")
    X_batch = tf.strings.split(X_batch)
    return X_batch.to_tensor(default_value = b"<pad>"), y_batch


# In[6]:


preprocess(X_batch, y_batch)


# ## Constructing the Vocabulary :

# In[7]:


from collections import Counter


# In[8]:


vocabulary = Counter()
for X_batch, y_batch in datasets["train"].batch(2).map(preprocess):
    for review in X_batch:
        vocabulary.update(list(review.numpy()))


# In[9]:


vocabulary.most_common()[:5]


# In[10]:


len(vocabulary)


# ## Truncating the Vocabulary :

# In[11]:


vocab_size = 10000
truncated_vocabulary = [words for words, count in vocabulary.most_common()[:vocab_size]]


# ## Creating a lookup table :

# In[12]:


words = tf.constant(truncated_vocabulary)
word_ids = tf.range(len(truncated_vocabulary), dtype = tf.int64)
vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
#print(vocab_init)


# In[13]:


num_oov_buckets = 1000
table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)


# In[14]:


table.lookup(tf.constant([b"This movie was faaaaaantastic".split()]))


# ## Creating the Final Train and Test sets :

# In[15]:


def encode_words(X_batch, y_batch):
    return table.lookup(X_batch), y_batch

train_set = datasets["train"].repeat().batch(32).map(preprocess)
train_set = train_set.map(encode_words).prefetch(1)


# In[16]:


test_set = datasets["test"].batch(1000).map(preprocess)
test_set = test_set.map(encode_words)


# In[17]:


for X_batch, y_batch in train_set.take(1):
    print(X_batch)
    print(y_batch)


# ## Building the Model :

# In[18]:


embed_size = 128
model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size,
                          mask_zero = True,
                          input_shape = [None]),
    keras.layers.GRU(4, return_sequences = True),
    keras.layers.GRU(2),
    keras.layers.Dense(1, activation = 'sigmoid')
])


# In[19]:


model.compile(loss = 'binary_crossentropy', optimizer="adam", metrics = ["accuracy"])


# ## Training and Testing the Model :

# In[20]:


import time
start = time.time()


# In[21]:


model.fit(train_set, steps_per_epoch=train_size//32, epochs=2)


# In[22]:


end=time.time()
print("Time of execution : ", end-start)


# In[23]:


model.evaluate(test_set)

