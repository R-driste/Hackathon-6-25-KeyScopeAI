import warnings
warnings.filterwarnings("ignore")
import nltk
nltk.download('punkt')

import pandas as pd
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Dropout

from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from nltk.corpus import stopwords

import csv
def read_csv_with_error_handling(file_path):
    rows = []
    with open(file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for i, row in enumerate(csv_reader):
            try:
                rows.append(row)
            except csv.Error as e:
                print(f"Error encountered at row {i + 1}: {str(e)}")
                continue
    df = pd.DataFrame(rows[1:], columns=rows[0])  
    return df

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

max_seq_length = 50  
vocab_size = 20000
embedding_dim = 100
lstm_units = 128
num_classes = 3

input1 = Input(shape=(max_seq_length,))
embedding1 = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input1)
lstm1 = Bidirectional(LSTM(units=lstm_units, return_sequences=True))(embedding1)
lstm1 = Dropout(0.2)(lstm1)

input2 = Input(shape=(max_seq_length,))
embedding2 = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input2)
lstm2 = Bidirectional(LSTM(units=lstm_units, return_sequences=True))(embedding2)
lstm2 = Dropout(0.2)(lstm2)

merged = Concatenate()([lstm1, lstm2])
lstm3 = Bidirectional(LSTM(units=lstm_units))(merged)
lstm3 = Dropout(0.2)(lstm3)

dense1 = Dense(units=64, activation='relu')(lstm3)
dense1 = Dropout(0.2)(dense1)
dense2 = Dense(units=32, activation='relu')(dense1)

output = Dense(units=num_classes, activation='softmax')(dense2)

model = Model(inputs=[input1, input2], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

early_stopping = EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def get_dev():
  df_dev = pd.read_csv("static/python/snli_1.0_dev.csv")
  return df_dev
def get_train():
  df_train = read_csv_with_error_handling("static/python/snli_1.0_train.csv")
  return df_train
def get_test():
  df_test = pd.read_csv("static/python/snli_1.0_test.csv")
  return df_test

df_dev = get_dev()
df_train = get_train()
df_test = get_test()
df_concatenated = pd.concat([df_dev, df_train], axis=0)
df_train = df_concatenated.reset_index(drop=True)
df_train = df_train[df_train['gold_label'] != '-']
x1_train,x2_train,y_train = df_train["sentence1_binary_parse"],df_train["sentence2_binary_parse"],df_train["gold_label"]
print('doneee')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_seq_length = 50  

tokenizer = Tokenizer(num_words=20000)  
tokenizer.fit_on_texts(x1_train + x2_train)  

x1_train = tokenizer.texts_to_sequences(x1_train)
x2_train = tokenizer.texts_to_sequences(x2_train)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

num_classes = len(label_encoder.classes_)
y_train = to_categorical(y_train, num_classes=num_classes)

x1_train = pad_sequences(x1_train, maxlen=max_seq_length, padding='post')
x2_train = pad_sequences(x2_train, maxlen=max_seq_length, padding='post')