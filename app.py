import blstm_model as m1
#import spacy_model as m2

print('\n')

for ind in range(14):

    #print('-------------------------------------- ' + str(ind) + ' -----------------------------------------')

    #m2.predict(ind, m2.test_data, m2.model)

    #print('\n-----------------------------------------------------------------------------------')

    #m1.predict(ind, m1.X_test, m1.y_test, m1.model, m1.idx2tok, m1.idx2ent)
    m1.predict(ind)

    #print('\n')


"""
sentenca_nova = "gostaria de uma casa no centro de sao paulo"
my_sent = ""
for ts in sentenca_nova:
    my_sent = my_sent + m1.idx2tok[ts] + " "
print('\n\nSENTENÇA NOVA:', my_sent, '\n\n')
"""
