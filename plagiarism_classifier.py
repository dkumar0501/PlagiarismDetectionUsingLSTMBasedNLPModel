import pandas as pd
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Subtract
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

import re
import nltk
import spacy
import seaborn as sns
from tqdm import tqdm
from tqdm.notebook import tqdm
from nltk.corpus import stopwords
from tqdm import tqdm

# Read data from text
with open('train_snli.txt', encoding='utf-8') as file:
    data = file.readlines()
    
# prepare csv file
with open('data.csv', 'w', newline= '') as csvfile:
    filenames = ['source_txt', 'plagiarism_txt', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=filenames)
    
    writer.writeheader()
    for line in tqdm(data):
        parts = line.strip().split('\t')
        source_txt = parts[0]
        plagiarishm_txt = parts[1]
        label = int(parts[2])
        
        writer.writerow({
            'source_txt' : source_txt,
            'plagiarism_txt' : plagiarishm_txt,
            'label' : label
        })
print('CSV file created successfully...')

# Replace with your dataset path
df = pd.read_csv("data.csv")
df.head()

df.duplicated().sum()

df.drop_duplicates(inplace=True)

import re, string


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text) #Replace newlines with spaces
    text = text.translate(str.maketrans("", "", string.punctuation)) #Remove punctuation
    return text


# Apply cleaning
df["source_clean"] = df["source_txt"].astype(str).apply(clean_text)
df["plag_clean"] = df["plagiarism_txt"].astype(str).apply(clean_text)

import sys
print(sys.executable)

# tokensiation

source = df['source_clean'].tolist()
plag = df['plag_clean'].tolist()
labels = df['label'].values


tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(source + plag)


source_seq = tokenizer.texts_to_sequences(source)
plag_seq = tokenizer.texts_to_sequences(plag)


max_len = 50
source_pad = pad_sequences(source_seq, maxlen=max_len)
plag_pad = pad_sequences(plag_seq, maxlen=max_len)

# split data

X_train_src, X_test_src, X_train_plag, X_test_plag, y_train, y_test = train_test_split(
source_pad, plag_pad, labels, test_size=0.2, random_state=42)


# Building LSTM Model
#Saimese LSTM network

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
lstm_dim = 64

input_a = Input(shape=(max_len,))
input_b = Input(shape=(max_len,))

embedding = Embedding(vocab_size , embedding_dim , input_length = max_len)
lstm = LSTM(lstm_dim)

encoded_a = lstm(embedding(input_a))
encoded_b = lstm(embedding(input_b))

merged = Subtract()([encoded_a , encoded_b])
merged = Dense(64 , activation = "relu")(merged)
merged = Dropout(0.5)(merged)
out = Dense(1,activation="sigmoid")(merged)

model = Model(inputs=[input_a , input_b], outputs=out)
model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
model.summary()


from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)


# model training
history = model.fit(
    [X_train_src, X_train_plag],
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

#evaluation

pred = model.predict([X_test_src, X_test_plag])
pred_label = (pred > 0.5).astype(int)


print("Accuracy:", accuracy_score(y_test, pred_label))
print(classification_report(y_test, pred_label))

# plotting training curve

plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy over epochs')
plt.legend()
plt.show()


plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss over epochs')
plt.legend()
plt.show()

# sample

idx = 0
src = X_test_src[idx:idx+1]
plg = X_test_plag[idx:idx+1]
print("Pred:", model.predict([src, plg])[0][0])

# test

def predict_pair(source_text, plag_text, tokenizer, max_len):
    # Convert to sequences
    src_seq = tokenizer.texts_to_sequences([source_text])
    plg_seq = tokenizer.texts_to_sequences([plag_text])
    
    # Pad
    src_pad = pad_sequences(src_seq, maxlen=max_len)
    plg_pad = pad_sequences(plg_seq, maxlen=max_len)

    # Predict
    pred = model.predict([src_pad, plg_pad])[0][0]

    # Show result
    print("Prediction score:", pred)
    if pred > 0.5:
        print("✅ Likely PLAGIARIZED")
    else:
        print("❌ Likely NOT plagiarized")

    return pred

src = "A woman is standing outside wearing glasses."
plg = "A woman wearing glasses is outdoors."

predict_pair(src, plg, tokenizer, max_len)