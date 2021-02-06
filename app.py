import blstm_model as m1
import spacy_model as m2

import graphics

import sys


# Opção para testar apenas sentenças abertas, fora do vocabulário (para o modelo BLSTM):
"""
while True:
    test_text = input("\nDigite a sentença para reconhecer as entidades treinadas (ou 'fim' pra sair):\n")
    if test_text == "fim":
        break
    m1.predict_any(test_text)

sys.exit()
"""


# Relacionados as metricas para  modelo BLSTM
accuracy_test_hist = []
f1score_test_hist = []

print('\n')

# Itera as 14 instâncias dos dados de treino:
for ind in range(14):
    
    #print('-------------------------------------- ' + str(ind) + ' -----------------------------------------')
    
    # Predição do modelo BLSTM:
    ac, f1 = m1.predict(ind)
    accuracy_test_hist.append(ac)
    f1score_test_hist.append(f1)
    
    # Predição do modelo SPACY:
    m2.predict(ind, m2.test_data, m2.model)
    
    #print('\n')


# grafico das métricas para o modelo BLSTM:
graphics.show_test_metric(accuracy_test_hist, 'accuracy')
graphics.show_test_metric(f1score_test_hist, 'f1-score')
