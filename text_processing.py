import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


accented = 'áàâãéèêẽíìîĩóòôõúùûũüç'
removed  = 'aaaaeeeeiiiioooouuuuuc'

def token_transform(sentence):
    """ Retorna sentença sem acentuações e em lowercase. Não altera o tamanho da sentença """

    sentence = sentence.lower()

    for x in range(len(accented)):
        sentence = sentence.replace(accented[x], removed[x])
    
    return sentence



chars_to_remove = [',', '.', ':', '!', '?']

def removes_useless_chars(df):
    """ Remove chars inúteis e espaço em branco no final de cada token da sentença. Recebe DataFrame com coluna 'Token' """

    for ind in range(len(df['Token'])):

        # Remove espaço em branco do final, se houver:
        df['Token'][ind] = df['Token'][ind].strip()
    
        # Remove caracter inútil do final, se houver:
        if df['Token'][ind][-1] in chars_to_remove:
            df['Token'][ind] = df['Token'][ind][0:-1]

    return df



def split_two_arrays(X, y):
    """ Separa os dados: 80% para treino e 20% para teste, com shuffle """
    return train_test_split(X, y, test_size=0.2, random_state=1)



def split_one_array(data):
    """ Separa os dados: 80% para treino e 20% para teste, com shuffle """
    return train_test_split(data, test_size=0.2, random_state=1)



def one_hot_enc(y, n_entities):
    """ Aplica OneHotEncoding em y """
    return [to_categorical(elem, num_classes=n_entities) for elem in y]



def print_elements(n, iter, title):
    """ Faz print de n elementos do iterável """

    print('\n{}:'.format(title))
    for count, it in enumerate(iter):
        if it[-1:] != '\n':
            print(it, '\n')
        else:
            print(it)
        if count == n-1:
            break


def print_lines(n, text, title):
    """  Faz print das n linhas do texto """

    print('\n{}:'.format(title))
    line = text.splitlines()
    for i in range(n+1):
        print(line[i])



def print_shapes(title, *arrays):
    """ Mostra os shapes dos dados de treino e teste """

    line = ''
    for arr in arrays:
        line = line + str(np.array(arr).shape) + ' | '
    line = line[0:-3]
    
    print('\n{}{}\n'.format(title, line))



def shape(arr):
    return str(np.array(arr).shape)



def print_sentence_indexes(n, X, y, idx2tok, idx2ent):
    """ Faz print da indexação dos tokens e labels das n sentenças """

    for i in range(n):
        print('## Sentença', i+1, '\n')

        print('---> Tokens:')
        print(X[i])
        print([idx2tok[j] for j in X[i]])
        print('')

        print('---> Labels:')
        print(y[i])
        print([idx2ent[j] for j in y[i]])
        print('')
    print('')


