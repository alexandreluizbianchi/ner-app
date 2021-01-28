import io
import os
import json
import pandas as pd
import numpy as np

#from tensorflow.keras import Model, Input
#from tensorflow.keras.layers import LSTM, Embedding, Dense
#from tensorflow.keras.layers import TimeDistributed, Dropout, Bidirectional

#import keras

"""
from keras import Model, Input, models
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.preprocessing.sequence import pad_sequences
"""
from tensorflow.keras import Model, Input, models, optimizers
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences



from sklearn.metrics import accuracy_score, f1_score

import text_processing as tp
import graphics



def get_entity(t_ind, sent, json_sent):
    """ Retorna a entidade convertida para o formato IOB (Inside, Outside, Beginning) """

    for annot in json_sent['labels']:
    
        if len(annot) == 3:
        
            if t_ind == annot[0]:
                return "B-"+annot[2]
            
            if t_ind > annot[0] and t_ind < annot[1]:
                return "I-"+annot[2]

    return "O"



def build_csv_dataset(filename):
    """ Constrói o conteúdo (str) do dataset CSV (ponto e vírgula) no formato IOB """

    # separador = ';' pela questão de conter tokens com vírgula (exemplo: moeda pode ter vírgula)
    csv_dataset = "Sentence;Token;Entity\n"

    with open(filename, mode="r", encoding="utf-8") as f:

        lines = f.readlines()

        # Show 5 sentences:
        tp.print_elements(5, lines, 'Dataset Original')

        for count, line in enumerate(lines):

            json_sent = json.loads(line)

            # Passa para lowercase e remove acentos:
            json_sent['text'] = tp.token_transform(json_sent['text'])

            # build CVS dataset
            position = 0
            tokens = json_sent['text'].split()
            for token in tokens:
            
                entity = get_entity(position, json_sent['text'], json_sent)
                csv_dataset += "sentence-"+str(count+1)+";"+token+";"+entity+"\n"

                position = position + len(token) + 1

    # Mostra 30 primeiras linhas:
    tp.print_lines(30, csv_dataset, 'Dataset CSV IOB')

    return csv_dataset



def build_model(sent_maxlen, n_tokens, n_entities):
    """ Constrói o modelo com arquitetura BLSTM """

    print('\n')

    input_word = Input(shape=(sent_maxlen,))
    #model = Embedding(input_dim=n_tokens, output_dim=sent_maxlen, input_length=sent_maxlen)(input_word)
    model = Embedding(input_dim=n_tokens, output_dim=130)(input_word)
    model = Dropout(0.6)(model)
    model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
    out = TimeDistributed(Dense(n_entities, activation="softmax"))(model)
    model = Model(input_word, out)
    
    model.summary()

    return model



def compile_model(model):
    """ Compila o modelo """

    opt = optimizers.Adam(lr=0.01, decay=1e-6)
    #opt = "adam"#lr default = 0.001
    loss = "categorical_crossentropy"

    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    #model.compile(optimizer=opt, loss=loss, metrics=["categorical_accuracy"])

    return model



def train_model(model, X_train, y_train):
    """ Treina o modelo com 20% dos dados para validação, e depois retorna o histórico """

    print('')
    return model.fit(X_train, np.array(y_train), batch_size=1, epochs=30, validation_split=0.2, verbose=1, shuffle=False)



def iob_to_dict(prediction, sentence, idx2ent):
    """ Faz conversão do formato IOB para JSON """

    ret = {}

    prediction = np.argmax(prediction, axis=-1)

    if len(prediction[0]) != len(sentence):
        print('Erro: prediction[0] and sentence do not have the same length')
        return ret
  
    current_text = ''
    current_label = ''

    for t, e in zip(sentence, prediction[0]):
    
        
        if idx2ent[e] == 'O':
            # outside
            current_label = ""
            current_text = ""
            continue
    
        parts = idx2ent[e].split('-')
        if parts[0] == 'B':
            # beginning:
            current_label = parts[1]
            current_text = tokens[t]
            ret[current_text] = current_label

        else:
            # inside (parts[0] == 'I')
            if parts[1] == current_label:
                # apaga item anterior e cria nova chave (current_text+tokens[t])
                ret.pop(current_text, None)
                current_text = current_text + ' ' + tokens[t]
                ret[current_text] = current_label

    return ret



#def predict(i, X_test, y_test, model, idx2tok, idx2ent):
def predict(i):
    """ Faz predição usando X_test[i] - dados de teste """

    p = model.predict(np.array([X_test[i]]))

    
    
    # Imprime sentença:
    #print('\n')
    my_sent = ""
    for ts in X_test[i]:
        my_sent = my_sent + idx2tok[ts] + " "
    #print(my_sent.strip(), '\n')

    """
    # Imprime no formato Spacy
    gen_dict = iob_to_dict(p, X_test[i], idx2ent)
    for k in gen_dict.keys():
        print("{:20} : {:20}".format(gen_dict[k], k))
    """



    p = np.argmax(p, axis=-1)
    yt = np.argmax(y_test[i], axis=1)

    p_label = []
    yt_label = []

    # comentar
    print("\n{:20}{:20}\t {}".format("Token", "True", "Pred"))
    # comentar
    print("-" *55)
    for t, true, pred in zip(X_test[i], yt, p[0]):
        # comentar
        print("{:20}{:20}\t{}".format(tokens[t], idx2ent[true], idx2ent[pred]))
        
        if tokens[t] != 'PADDING':
            p_label.append(idx2ent[pred])
            yt_label.append(idx2ent[true])

    """
    ## Acuracia: {} | F1-Score macro: {}\n'.format(accuracy_score(yt_label, p_label), 
        f1_score(yt_label, p_label, average='macro')))
    """

    col1 = '## Acuracia: {}'.format(accuracy_score(yt_label, p_label))
    col2 = 'F1-Score macro: {}'.format(f1_score(yt_label, p_label, average='macro'))
    print('{:40}|\t{:40} --> {}'.format(col1, col2, my_sent[:80]))





# Constrói dataset CSV IOB com pré-processamento (conversão de acentos e maiúsculos)
csv_dataset = build_csv_dataset("dataset_22_01.json")


# Gera DataFrame
df_csv_dataset = pd.read_csv(io.StringIO(csv_dataset), sep=';', encoding="latin1")
print('\nDataFrame Original:')
print(df_csv_dataset.head(30))


# Limpa tokens
df_csv_dataset = tp.removes_useless_chars(df_csv_dataset)
print('\nDataFrame Sem Pontuações:')
print(df_csv_dataset.head(30))


# Define variáveis de tokens e entidades únicos
tokens = sorted(list(set(df_csv_dataset['Token'].values)))
tokens.append('PADDING')
n_tokens = len(tokens)
entities = sorted(list(set(df_csv_dataset['Entity'].values)))
n_entities = len(entities)
print('\n-->', n_tokens, 'tokens  :', tokens)
print('\n-->', n_entities, 'entities:', entities, '\n')



# Define lista de sentenças agrupadas e algumas variáveis
agg_func = lambda s: [(t, e) for t, e in zip(s["Token"].values.tolist(),s["Entity"].values.tolist())]
grouped = df_csv_dataset.groupby("Sentence").apply(agg_func)
sentences = [s for s in grouped]
sent_len = [len(sent) for sent in sentences]
sent_maxlen = max(sent_len)

#sent_maxlen = 30# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


print('\nExploração rápida do dataset:')
print('\n--> Sentença 1:', sentences[0])
print('\n--> Sentença 2:', sentences[1])
print('\n--> Sentença 3:', sentences[2])
print('\n--> Sentença 4:', sentences[3])
print('\n--> Quantidade de sentenças:', len(sentences))
print('\n--> Qtde de tokens de cada sentença:', sent_len)
print('\n--> Qtde máxima de tokens:', sent_maxlen, '\n')


"""
# Visualizações:
graphics.show_hist_token_sizes(sent_len)
graphics.show_hist_classes(df_csv_dataset)
"""


# Define dicionários para conversões de/para índices
tok2idx = {t: i for i, t in enumerate(tokens)}
ent2idx = {e: i for i, e in enumerate(entities)}
idx2tok = {i: t for t, i in tok2idx.items()}
idx2ent = {i: e for e, i in ent2idx.items()}



# Define vetores de entrada (palavras e labels) como X e y
X = [[tok2idx[pair[0]] for pair in sent] for sent in sentences]
y = [[ent2idx[pair[1]] for pair in sent] for sent in sentences]

# Mostra indexações para 2 sentenças
print('\nVerificação da indexação dos tokens e entities ANTES do padding:')
tp.print_sentence_indexes(2, X, y, idx2tok, idx2ent)



# Realiza o padding com o tamanho máximo igual ao maior vetor (sent_maxlen), para X e y

# truncating="post" (corta no final se sentença for maior que sent_maxlen)

X = pad_sequences(maxlen=sent_maxlen, sequences=X, padding="post", value=tok2idx["PADDING"])
y = pad_sequences(maxlen=sent_maxlen, sequences=y, padding="post", value=ent2idx["O"])

# Mostra novamente as indexações para 2 sentenças (com padding)
print('\nVerificação da indexação dos tokens e entities DEPOIS do padding:')
tp.print_sentence_indexes(2, X, y, idx2tok, idx2ent)

# Mostra (check) nova dimensão dos vetores X e y
print('\nVerificação da nova dimensão dos vetores X e y após o padding:')
print('X:', [len(sz) for sz in X])
print('y:', [len(sz) for sz in y])



# Aplica One Hot Encoding nos labels (que não passarão pela camada de Embedding)
print('\nDimensões dos labels (y) antes do OneHot Encoding:')
print([elem.shape for elem in y])
y_enc = tp.one_hot_enc(y, n_entities)
print('\nDimensões dos labels (y) depois do OneHot Encoding:')
print([elem.shape for elem in y_enc])


# Separa dados de Treino e dados de Testes (shuffle=True)
X_train, X_test, y_train, y_test = tp.split_two_arrays(X, y_enc)

# Exibe os shapes:
print('\nSeparação de X e y em dados de treino e teste:')
print('Dimensões: X_train = {} | y_train = {} | X_test = {} | y_test = {}'.
    format(tp.shape(X_train), tp.shape(y_train), tp.shape(X_test), tp.shape(y_test)))


#if not os.path.exists('BLSTM_model.h5'):
if True:

    # Constrói e compila o modelo, exibindo seu resumo
    model = build_model(sent_maxlen, n_tokens, n_entities)
    model = compile_model(model)

    # Treina o modelo com 20% dos dados para validação
    history = train_model(model, X_train, y_train)


    # Salva modela treinado
    #models.save_model(model, "BLSTM_model.h5", overwrite = True, include_optimizer = False)#False

else:

    #model = keras.models.load_model("BLSTM_model.h5", compile = False)
    model = models.load_model("BLSTM_model.h5", compile = False)#False
    #model = compile_model(model)



# Exibe resultados do treino
graphics.show_training_metric(history, 'accuracy')
graphics.show_training_metric(history, 'loss')



"""
# Faz predições com dados de teste
index = 13
predict(index, X_test, y_test, model, idx2tok, idx2ent)
"""
