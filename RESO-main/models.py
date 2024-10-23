"""
Funciones relacionadas al entrenamiento de modelos
"""

from tabulate import tabulate
import fasttext, re


# Funciones para entrenamiento supervisado
# =============================================================================
def train_and_test(train_fname, test_fname, epochs, lrate=0.1, wng=1, dim=100,
                   ws=5, loss_fn='hs'):
    """
    Entrena un modelo de FastText e imprime por pantalla los resultados
    del testeo
    Args:
        train_fname   :   Nombre del archivo con los datos de entrenamiento.   
        test_fname    :   Nombre del archivo con los datos de testeo.
        lrate         :   Learning rate del optimizador.  [Default 0.1]
        wng           :   Tamaño maximo del ngrama.       [Default 1]
        dim           :   Dimension del word embedding.   [Default 100]
        ws            :   Tamaño de la context window.    [Default 5]
        loss_fn       :   Funcion de costo del entrenamiento. Por defecto se utiliza
                          Hierarchical Softmax, el cual es una aproximacion numerica 
                          eficiente de Softmax
    Returns:
        model         :   Modelo de FastText entrenado.a
        test_results  :   Presicion y recall obtenidos en el conjunto de testeo 
    """

    # Definicion del modelo
    model = fasttext.train_supervised(
        input = train_fname, 
        lr    = lrate,
        dim   = dim,
        ws    = ws, 
        epoch = epochs, 
        loss  = loss_fn,
        verbose = 0,    # Evita los prints de progreso de FastText
        wordNgrams = wng,
    )

    # Testeo del modelo
    test_results = list(model.test(test_fname))

    # Print de resultados
    headers = ["Cantidad de samples", "Presicion", "Recall"]
    print(tabulate([test_results], headers=headers, tablefmt="pretty"))

    return model, test_results[1:]


def predict(model, sample):
    """
    FastText cuando predice da como resultado un string con muchas
    labels, esta funcion recupera la prediccion mas probable 
    Args:
        model:           Modelo entrenado
        sample:          Sample de interes
    Return:
        prediction:      Label predicha por el modelo
    """
    pattern = "__label__(.*)__"
    prediction =  re.findall(pattern, model.predict(sample)[0][0])[0]

    try:
        prediction = int(prediction)
    except ValueError:
        # Si no se puede parsear a un entero es porque la prediccion es 
        # una string
        pass
    return prediction


# Funciones relacionadas a metricas de los modelos
# =============================================================================
def test_and_print(model, test_fname):
    """
    Evalua el modelo 'model' sobre el conjunto de testeo almacenado en el archivo
    'test_fname' e imprime los resultados en una tabla ASCII

    Args:
        model         :  Modelo FastText entrenado
        test_fname    :  Nombre del archivo que contiene los datos de testeo
    """
    # Testeo del modelo
    test_results = list(model.test(test_fname))
    # Printeo de resultados
    headers = ["Cantidad de samples", "Presición", "Recall"]
    print(tabulate([test_results], headers=headers, tablefmt="pretty"))


def print_metricas(model, test_fname):
    """
    Imprime las obtenidas por un modelo de FastText sobre un conjunto
    de testeo y las imprime por pantalla. Las metricas se calculan por label, las
    metricas contempladas son:
        + Presicion
        + Recall
        + F1-Score
    Args:
        model         :     Modelo FastText entrenado
        test_fname    :     Nombre del archivo de texto del conjunto de testeo
    """
    labels = model.get_labels()                 # Labels del modelo
    metricas = model.test_label(test_fname)     # Metricas para cada label

    data = []                       # Lista para recopilar los resultados
    pattern = "__label__(.*)__"     # Patron que captura el nombre de la label
    for label in labels:    
        label_name = re.findall(pattern, label)[0]      # Nombre de la label
        label_presicion = metricas[label]['precision']  # Presicion para la label
        label_recall    = metricas[label]['recall']     # Recall para la label
        label_f1        = metricas[label]['f1score']    # F1-Score para la label
        data.append([label_name, label_presicion, label_recall, label_f1])

    headers = ["Label", "Presicion", "Recall", "F1"]
    # Imprimimos una tabla con los resultados
    print(tabulate(data, headers=headers, tablefmt="pretty"))