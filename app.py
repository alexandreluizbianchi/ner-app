import blstm_model as m1
import spacy_model as m2

import sys

"""
while True:
    test_text = input("\nDigite a sentença para reconhecer as entidades treinadas (ou 'fim' pra sair):\n")
    if test_text == "fim":
        break
    m1.predict_any(test_text)

sys.exit()
"""


print('\n')
for ind in range(14):
    #print('-------------------------------------- ' + str(ind) + ' -----------------------------------------')
    m1.predict(ind)
    m2.predict(ind, m2.test_data, m2.model)
    #print('\n')

