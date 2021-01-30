
""" TESTE 1 --------------------------------------------------------

from sklearn.metrics import accuracy_score, f1_score

yt_label = ['B-tipo_imovel', 'B-local_ref', 'I-local_ref', 'I-local_ref', 'O', 'B-cidade', 'I-cidade', 'O', 'O', 'O', 'O', 'B-vlr_max_imovel', 'O', 'O']
p_label = ['O', 'B-tipo_imovel', 'I-local_ref', 'I-local_ref', 'O', 'B-cidade', 'I-local_ref', 'O', 'O', 'O', 'O', 'B-vlr_max_imovel', 'O', 'O']

print('## Acuracia - vetores categoricos:', accuracy_score(yt_label, p_label), '\n')

# macro-média não leva em conta o desequilíbrio da classe 
# e a pontuação f1 da classe A será tão importante quanto a pontuação f1 da classe B
print('## F1-macro    - vetores categoricos:', f1_score(yt_label, p_label, average='macro'))
#print('## F1-micro    - vetores categoricos:', f1_score(yt_label, p_label, average='micro'))
#print('## F1-weighted - vetores categoricos:', f1_score(yt_label, p_label, average='weighted'))
#print('## F1-None     - vetores categoricos:', f1_score(yt_label, p_label, average=None), '\n')

texto = 'aaaaaaaaabbbbbbbbcccccccddddddddddddeeeeeeeeeeeffffffffffggggghhhhhh9'

print('{}'.format(texto[:80]))

"""


""" TESTE 2 --------------------------------------------------------------

import os

if os.path.exists('BLSTM.model'):
    print('Sim') 
else:
    print('Não')

"""


""" TESTE 3 ----------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras import layers

embedding_layer = layers.Embedding(500, 5)

#result = embedding_layer(tf.constant(['ABC-456', 'DEF', 'GHIJK']))
result = embedding_layer(tf.constant([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))

print(result.numpy())
"""


""" TESTE 4 ----------------------------------------------------------------
from tensorflow.keras.preprocessing.sequence import pad_sequences

X = [[1, 2, 3, 4, 5, 6, 7]]
print("X antes:", X)

X = pad_sequences(maxlen=3, sequences=X, padding="post", value=99)
print("X depois:", X)
"""


to_index = {'marte':0, 'terra':1, 'venus':2, 'jupiter':3, 'mercurio':4, 'saturno':5}

lst = ['terra', 'saturno', 'plutao', 'marte', 'sol', 'venus', 'lua']

for t in lst:

    if t not in to_index.keys():
        next_i = to_index[list(to_index.keys())[-1]] + 1
        to_index[t] = next_i

    print(t, ':', to_index[t])



