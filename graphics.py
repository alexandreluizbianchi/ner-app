import matplotlib.pyplot as plt
import numpy as np


def show_hist_token_sizes(size_array):
    """ Exibe histograma com as quantidades de tokens em todas as sentenças """

    #print(sorted(size_array))

    _, ax = plt.subplots(figsize=(15,5))
    ax.set_xlabel("Quantidade de tokens em cada sentença")
    ax.set_title("Distribuição das quantidades de tokens")
    plt.xticks(rotation=45)
    mybins=np.arange(min(size_array)-1, max(size_array)+2, 1)
    plt.xticks(mybins)
    plt.hist(size_array, bins=mybins, align='left')
    plt.show()



def show_hist_classes(df):
    """ Exibe histograma com todas as classes (entidades) """

    _, ax = plt.subplots(figsize=(15,5))
    ax.set_xlabel("Entidades")
    ax.set_title("Distribuição de Classes")
    plt.xticks(rotation=45)
    en = [t for t in df['Entity'] if t != 'O']
    lst_ents = list(set(en))

    plt.hist(en, bins=5*len(lst_ents), align='mid')

    plt.show()



def show_training_metric(history, metric):
    """ Exibe um gráfico da metrica desejada (treino e validação) """

    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(['train', 'validation'])
    plt.show()


def show_test_metric(metric_lst, metric):
    """ Exibe um gráfico da metrica desejada para os testes """

    plt.plot(metric_lst)
    plt.xlabel("Instances")
    plt.ylabel(metric)
    plt.legend(['test'])
    plt.show()

