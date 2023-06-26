import static.python.contradictions as contradictions
from contradictions import tokenizer, pad_sequences
from contradictions import max_seq_length
from contradictions import model

model.load_weights('model_weights.h5')

def aiprocess(data):
    datafocus = data[-1]
    datafocus = tokenizer.tokenize(datafocus)
    datafocus = pad_sequences((datafocus), maxlen=max_seq_length, padding='post')
    for statement in data[:-2]:
        evaluation = model.predict((statement,datafocus))
        if evaluation==[1., 0., 0.]:
            print('ya')
            return "orangeflag"
        print('a')
    print('na')
    return "greenflag"

aiprocess(['hi','bye','no'])