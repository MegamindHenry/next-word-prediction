from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)


def print_full_prop_list(model, tokenizer, context, sort=True):
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    encoded = tokenizer.texts_to_sequences([context])[0]
    encoded = pad_sequences([encoded], maxlen=4, truncating='pre')

    yhat_row = model.predict(encoded)[0]

    sort_index = np.argsort(yhat_row)[::-1]

    for x in sort_index:
        if x == 0:
            print("UNK", yhat_row[x])
        else:
            print(reverse_word_map[x], yhat_row[x])


# load cleaned text sequences
in_filename = 'republic_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1

# load the model
model = load_model('model.h5')

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))

# #select a seed text
# seed_text = lines[randint(0, len(lines))]
# print(seed_text + '\n')
#
# # generate new text
# generated = generate_seq(model, tokenizer, seq_length, seed_text, 50)
# print(generated)

context = ["polemarchus polemarchus polemarchus polemarchus"]
context1 = ["me said to me"]
context2 = ["polemarchus i to me"]
print("----------111111----------")
print_full_prop_list(model, tokenizer, context)
print("----------222222----------")
print_full_prop_list(model, tokenizer, context1)
print("----------333333----------")
print_full_prop_list(model, tokenizer, context2)