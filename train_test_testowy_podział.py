import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split


data=pd.read_csv('sincos_test.csv')

# tokens - top_time model
vocab_top = ['-p', '+p', '-l', '+l', 'd']
vocab_seq = ['1', '2', '3', '4']
vocab = {}
token_id = 0

for t in vocab_top:
    for s in vocab_seq:
        fused = f'{t}_{s}'
        vocab[fused] = token_id
        token_id += 1


def tokenize_sequence(token, vocab):
    tokenized = []
    for i in range(3):  # Since sequences are exactly length 3
        fused_key = token[i]
        tokenized.append(vocab[fused_key])
    return tokenized


# Tokenize data
data['splited_top'] = data['top'].apply(lambda text: re.findall(r'[+-][pl]|d', str(text)))
data['splited_seq'] = data['seq'].apply(lambda text: re.findall(r'1|2|3|4', str(text)))

data['tokens'] = data.apply(lambda row: [f'{top}_{seq}' for top, seq in zip(row['splited_top'], row['splited_seq'])],
                            axis=1)
data['tokenased'] = data.apply(lambda row: tokenize_sequence(row['tokens'], vocab), axis=1)
data=data.drop(['top','seq','tokens', 'splited_top', 'splited_seq', 'lay', 'copy'], axis=1)

#print(data)

train, test = train_test_split(data, test_size=0.01)
#print(test)
test_fin = data[data['tokenased'].isin(test['tokenased'])].reset_index(drop=True)
train_fin = train[~train['tokenased'].isin(test_fin['tokenased'])].reset_index(drop=True)

print(train_fin)
print(test_fin)