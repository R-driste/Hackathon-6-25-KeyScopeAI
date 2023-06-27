from static.python.contradictions import max_seq_length, model, read_csv_with_error_handling
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def get_dev():
  df_dev = pd.read_csv("static/python/snli_1.0_dev.csv")
  return df_dev
def get_train():
  df_train = read_csv_with_error_handling("static/python/snli_1.0_train.csv")
  return df_train
def get_test():
  df_test = pd.read_csv("static/python/snli_1.0_test.csv")
  return df_test

#uploading data from csv files
df_dev = get_dev()
df_train = get_train()
df_test = get_test()

#splitting data
df_concatenated = pd.concat([df_dev, df_train], axis=0)
df_train = df_concatenated.reset_index(drop=True)
df_train = df_train[df_train['gold_label'] != '-']
x1_train,x2_train,y_train = df_train["sentence1"],df_train["sentence2"],df_train["gold_label"]

#tokenizer creation and use
tokenizer = Tokenizer(num_words=20000)  
tokenizer.fit_on_texts(x1_train + x2_train)  
x1_train = tokenizer.texts_to_sequences(x1_train)
x2_train = tokenizer.texts_to_sequences(x2_train)

#padding x trains
x1_train = pad_sequences(x1_train, maxlen=max_seq_length, padding='post')
x2_train = pad_sequences(x2_train, maxlen=max_seq_length, padding='post')

#y encoding
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

num_classes = len(label_encoder.classes_)
y_train = to_categorical(y_train, num_classes=num_classes)

#loading model weights before usage
model.load_weights('static/python/model_weights_4.h5')

#storing tokenized sentences in memory
preprocessed = []
audios = []

#ai process function
def aiprocess(data):
    global preprocessed
    global audios
    #grab recent audio, tokenize and pad
    recentaudio = data[-1]
    audios.append(recentaudio)
    datafocus = recentaudio.split()
    datafocus = tokenizer.texts_to_sequences([data[-1]]) #in list
    datafocus = pad_sequences(datafocus, maxlen=max_seq_length, padding='post') #on its own
    preprocessed.append(datafocus)
    audio_index = 0
    #runs after second audio clip
    for statement in preprocessed[:len(preprocessed)-1]:
        evaluation = model.predict([statement,datafocus])
        print(evaluation)
        old_array = evaluation[0]
        print(f"for {recentaudio} and {audios[audio_index]}")
        if old_array[0] > old_array[1] and old_array[0] > old_array[2]:
            if ((1 - old_array[0]) < 0.3):
              audio_index += 1
              return(f'Issue for "{audios[audio_index-1]}" vs "{recentaudio}"')
            else:
              audio_index += 1
              return(f'Potential Issue, caution, for "{audios[audio_index-1]}" vs "{recentaudio}"')
        else:
            print('Checkpoint, safe')
        audio_index += 1  

    print('No Issue')
    return "greenflag"

def clearlist():
   global preprocessed
   global audios
   preprocessed = []
   audios = []
   return preprocessed, audios
