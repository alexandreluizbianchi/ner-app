import graphics
import sys


"""
# Opção 1 para testar apenas sentenças abertas, fora do vocabulário (para o modelo BLSTM):
import blstm_model as m1
while True:
    test_text = input("\nDigite a sentença para reconhecer as entidades treinadas (ou 'fim' pra sair):\n")
    if test_text == "fim":
        break
    m1.predict_any(test_text)
sys.exit()
"""


"""
# Opção 2 para testar apenas sentenças abertas, fora do vocabulário (para o modelo SPACY):
import spacy_model as m2
m2.user_predictions(m2.model)
sys.exit()
"""


# Opção 3 para comparar os modelos BLSTM e SPACY, sobre os 14 dados de teste (com metricas para modelo BLSTM)
import blstm_model as m1
import spacy_model as m2

accuracy_test_hist = []
f1score_test_hist = []

print('\n')

# Itera as 14 instâncias dos dados de teste:
for ind in range(14):
    
    #print('-------------------------------------- ' + str(ind) + ' -----------------------------------------')
    
    # Predição do modelo BLSTM:
    ac, f1 = m1.predict(ind)
    accuracy_test_hist.append(ac)
    f1score_test_hist.append(f1)
    
    # Predição do modelo SPACY:
    m2.predict(ind, m2.test_data, m2.model)

# grafico das métricas para o modelo BLSTM:
graphics.show_test_metric(accuracy_test_hist, 'accuracy')
graphics.show_test_metric(f1score_test_hist, 'f1-score')
