from nltk.util import pad_sequence
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.lm.preprocessing import flatten


import os
import requests
import io #codecs

try: # Use the default NLTK tokenizer.
    from nltk import word_tokenize, sent_tokenize
    # Testing whether it works.
    # Sometimes it doesn't work on some machines because of setup issues.
    word_tokenize(sent_tokenize("This is a foobar sentence. Yes it is.")[0])
except: # Use a naive sentence tokenizer and toktok.
    import re
    from nltk.tokenize import ToktokTokenizer
    # See https://stackoverflow.com/a/25736515/610569
    sent_tokenize = lambda x: re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', x)
    # Use the toktok tokenizer that requires no dependencies.
    toktok = ToktokTokenizer()
    word_tokenize = word_tokenize = toktok.tokenize


# Text version of https://kilgarriff.co.uk/Publications/2005-K-lineer.pdf
if os.path.isfile('language-never-random.txt'):
    with io.open('language-never-random.txt', encoding='utf8') as fin:
        text = fin.read()
else:
    url = "https://gist.githubusercontent.com/alvations/53b01e4076573fea47c6057120bb017a/raw/b01ff96a5f76848450e648f35da6497ca9454e4a/language-never-random.txt"
    text = requests.get(url).content.decode('utf8')
    with io.open('language-never-random.txt', 'w', encoding='utf8') as fout:
        fout.write(text)

tokenized_text = [list(map(str.lower, word_tokenize(sent)))
                  for sent in sent_tokenize(text)]


n = 3
train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)

from nltk.lm import WittenBellInterpolated
model = WittenBellInterpolated(n)

model.fit(train_data, padded_sents)

print(model.logscore("never", "language is".split()))
print(model.logscore("am", "language is".split()))
print(model.logscore("a", "language is".split()))
print(model.logscore("the", "language is".split()))
