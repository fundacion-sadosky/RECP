"""
Funciones relacionadas con la creacion de datasets
"""

import numpy as np
import re

from fasttext import tokenize
from normalizer import preprocesar_codigo, tokenizar_operadores

def shuffle_and_get_dataset(data, target, val_prop=0.2, test_prop=0.2, seed=191919):
    """
    Aleatoriza el dataset y reserva una porcion de los datos para validacion 
    y para testeo
    Args:
        data       :    Samples del dataset
        target     :    Targets correspondientes a cada sample
        val_prop   :    Proporcion de datos reservados para validacion. Por defecto 0.2
        test_prop  :    Proporcion de datos reservados para testeo. Por defecto 0.2
        seed       :    Seed para aleatorizar el dataset. Default(191919)
    Returns:
        (train_data, train_target)  :   Dataset y target de entrenamiento
        (val_data, val_target)      :   Dataset y target de validacion
        (test_data, test_target)    :   Dataset y target de testeo
    """

    # Aleatorizamos el dataset
    permutation = np.random.RandomState(seed).permutation(len(target))
    data   = data[permutation]
    target = target[permutation]

    # Reservamos una porcion de los datos y creamos los datasets de validacion 
    # y prueba
    val_size  = int(len(target)*val_prop)
    test_size = int(len(target)*test_prop) 

    val_data   = data[0: val_size]
    val_target = target[0: val_size]

    test_data   = data[val_size: val_size + test_size]
    test_target = target[val_size: val_size + test_size]

    train_data   = data[val_size+test_size :]
    train_target = target[val_size + test_size :]

    return (train_data, train_target), (val_data, val_target),\
            (test_data, test_target)


def es_significativo(code, target):
    """
    Estima si el codigo dado contiene una solucion significativa (i.e. que la intencion de
    code realmente sea la de un codigo y no un texto trivial), este filtro requiere 
    saber que tipo del codigo (si compila o no) para poder aplicar el 
    filtro apropiadamente
    Args:
        code    :  Code snippet
        target  :  submission_status asociado a code
    Returns:
        def_significativa : Booleano que indica si data contiene informacion
                            signficativa
    """
    if target == "errored":
        # Caso en que el codigo no compila
        return len(tokenize(preprocesar_codigo(code))) > 10
    else:
        # Caso en que el codigo compila y podemos usar regex
        pattern = r'\{.*\}' #r'\{(.*?\{*?.*)\}'
        matches = re.findall(pattern, code, re.DOTALL)
        def_significativa = False # Flag para detectar si hay una def significativa    
        for match in matches:
            # Buscamos una definicion signficativa entre las definiciones halladas
            match = tokenizar_operadores(match)
            if len(tokenize(match)) >= 5:
                return True
        return def_significativa
    

def dataset_to_file_norm(data, target, fname, normalizer):
    """
    Normaliza los datos en el dataset y usa los resultados junto a las labels 
    respectivas para crear un archivo de texto en un formato apropiado para entrenar un
    clasificador de FastText. 
    Es importante destacar que esta funcion asume que los elementos de data son listas 
    (ya que puede ser deseable considerar mas informacion ademas del codigo) y que 
    siempre en la primera posicion contienen codigo, ya que esta funcion usa un filtro
    de codigo trivial y espera codigo en esa posicion.  

    Args:
        data        :   Dataset
        target      :   Targets del dataset
        fname       :   Nombre del archivo resultante
        normalizer  :   Funcion de normalizacion apropiada para el tipo de los
                        elementos de data
    """
    with open(fname, "w") as file:
        for i in range(len(data)):
            # Filtramos las muestras con codigo trivial o no informativo
            if es_significativo(data[i][0], target[i]):
                code = normalizer(data[i]) 
                file.write(f"__label__{str(target[i])}__ {code}\n")