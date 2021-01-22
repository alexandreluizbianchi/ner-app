import sys
import json
import random

import spacy
from spacy.util import minibatch, compounding
import warnings

import text_processing as tp



def build_spacy_dataset(filename):
    """ Constrói o dataset Spacy (JSON-like) """

    # SPACY JSON format
    spacy_dataset = []

    with open(filename, mode="r", encoding="utf-8") as f:

        lines = f.readlines()
        for line in lines:

            json_sent = json.loads(line)

            # Passa para lowercase e remove acentos:
            json_sent['text'] = tp.token_transform(json_sent['text'])

            # build Spacy dataset
            internal_ents = []
            for annot in json_sent['labels']:
                if len(annot) == 3:
                    internal_ents.append((annot[0], annot[1], annot[2])) # (ini_ind,  end_ind,  entity)
            
            spacy_dataset.append((json_sent['text'], {'entities':internal_ents})) # (sentence, dict of entities)
    
    return spacy_dataset



def print_shapes(train_data, test_data):
    """ Mostra os shapes dos dados de treino e teste """

    print('train_data: Shape =', np.array(train_data).shape)
    print('test_data: Shape =', np.array(test_data).shape)



def build_train_model(train_data, epochs=32):
    """ Constrói e treina o modelo NER do Spacy com as épocas desejadas """

    #epochs = 32# Para uso de minibatches
    #epochs = 30# Sem minibatch (um registro de cada vez)

    nlp = spacy.blank('pt')  # create blank Language class

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)

    # add labels
    for _, annotations in train_data:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    # only train NER
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():

        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        # reset and initialize the weights randomly – (training a new model)
        optimizer = nlp.begin_training()

        for it in range(epochs):

            print("Starting epoch " + str(it))
            random.shuffle(train_data)
            losses = {}

            
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            
            #indice = 0
            for batch in batches:
                #indice += 1
                texts, annotations = zip(*batch)
                #print('tamanho dos batches:', len(texts), ' ou ', len(annotations))
                nlp.update(
                            texts,  # batch of texts
                            annotations,  # batch of annotations
                            drop=0.2,  # dropout - make it harder to memorise data
                            sgd=optimizer,  # callable to update weights
                            losses=losses,
                        )
                
            
            #print('Quantidade de batches:', indice)
            
            """
            for text, annotations in train_data:
                    nlp.update(
                        [text],  # batch of texts
                        [annotations],  # batch of annotations
                        drop=0.2,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
                        losses=losses)
            """
            print("Losses:", losses)

    return nlp



def predict(i, test_data, model):
    """ Faz a predição utilizando test_data[i] - dados de teste """


    # TODO
    # ====> USAR o test_data[i][1] RESPOSTA VERDADEIRA para comparação


    print(test_data[i][0], '\n')
    doc = model(test_data[i][0])
    for ent in doc.ents:
        print("{:20} : {:20}".format(ent.label_, ent.text))



def user_predictions(model):
    """ Teste interativo com dados inseridos pelo usuário """

    while True:
        test_text = input("\nDigite a sentença para reconhecer as entidades treinadas (ou 'fim' pra sair):\n")
        if test_text == "fim":
            break

        doc = model(test_text)

        for ent in doc.ents:
            print('==>', ent.text, ent.start_char, ent.end_char, ent.label_)




# Constrói o dataset para o Spacy
spacy_dataset = build_spacy_dataset("dataset_18_01.json")


# Separa 20% dos dados para teste (com shuffle)
train_data, test_data = tp.split_one_array(spacy_dataset)


# Exibe os shapes
print_shapes(train_data, test_data)


# Treina o modelo NER do Spacy
model = build_train_model(train_data, epochs=32)


# Faz a predição com os dados de teste
index = 0
predict(index, test_data, model)
