"""
Coleccion de funciones auxiliares para analizar modelos y para
analizar datasets
"""

import matplotlib.pyplot as plt
import numpy as np
import csv        

from sklearn.manifold      import TSNE
from sklearn.decomposition import PCA

# Funciones para testear robustez de modelos
# =============================================================================
def ocultar_nombre_funcion(code):
    """
    Reemplaza los nombres de las funciones que aparecen en code por el nombre "x"
    Args:
        code    :   Texto del codigo
    """
    try:
        code = code.split()
        for i in range(len(code)):
            if code[i] == "function":
                code[i+1] = "x"
        return " ".join(code)

    except IndexError:
        code.append("x")
        return " ".join(code)
    

def truncar_codigo(code, proporcion=0.2):
    """
    Trunca los ultimos tokens (determinados por espacios) en code
    Args:
        code:           Texto del codigo
        proporcion:     Porcentaje de tokens a truncar ([0, 1], default 0.2)
    """
    code = code.split()
    code = code[0: int(len(code) * (1 - proporcion))]
    return " ".join(code)


# Funciones para analizar datasets
# =============================================================================
def frecuencia_datos(target, xlabel, ylabel, title, fsize=(15, 6), xaxis=True):
    """
    Realiza un histograma de las categorias encontradas en el conjunto de labels
    y lo imprime por pantalla
    Args:
        target  :  Conjunto de etiquetas
        xlabel  :  Texto del eje horizontal
        ylabel  :  Texto del eje vertical
        title   :  Titulo del grafico
        fsize   :  Dimensiones de la imagen en pulgadas (ancho, largo)
        xaxis   :  Flag que determina si ocultar o no las notaciones sobre el
                   eje x, por defecto las muestra (True)
    """
    # Todas las categorias que aparecen en el dataset
    keys = np.sort(np.unique(target))   
    distribucion = {}

    # Conteo para cada categoria categoria
    for i in range(len(target)):
        try:
            distribucion[target[i]] += 1
        except KeyError:
            distribucion[target[i]] = 0


    # Especificacion del histograma
    plt.figure(figsize=fsize)
    plt.bar([str(key) for key in keys], 
            [np.count_nonzero(target == key) / len(target) for key in keys], 
            color="maroon")
    
    # Formateo del histograma
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if not xaxis:
        plt.xticks(ticks=[])  # Quita las divisiones del eje horizontal

    plt.show()
    return None


# Funciones para analizar modelos
# =============================================================================
def plot_confusion_matrix(predictions, targets):
    """
    Dibuja la matriz de confusion del modelo sobre los datos de entrenamiento 
    Args:
        predictions    :    Predicciones realizadas por el modelo
        targets        :    Labels reales de tales predicciones      
    """
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cm = confusion_matrix(targets, predictions)   # Matriz de confusion del modelo
    categorias = np.sort(np.unique(targets))      # Categorias posibles dentro de la prediccion

    display = ConfusionMatrixDisplay(cm, display_labels=categorias)
    display.plot()


# Funciones para analizar clusters
# =============================================================================
def plot_clustering_tsne(muestras, fname, clustering_map, labels_map, 
                         componentes=2, perplexity=50):
    """
    Realiza una reduccion de dimensionalidad de los datos proporcionados, 
    usando el metodo T-distributed Stochastic Neighbor Embedding y dibuja
    el resultado de esta reduccion y lo guarda en un archivo
    Args:
        muestras        :  Array de vectores a reducir dimensionalidad
        fname           :  Nombre de la imagen resultante
        clustreing_map  :  Mapeo de las muestras a cada cluster.
        labels_map      :  Mapeo de las muestras a cada label.
        componentes     :  Dimension resultante. Default [2]
        perplexity      :  Parametro relacionado a algoritmos de manifold learning.
                           Default [50].
    """
    # Reductor TSNE
    tsne_obj = TSNE(
        n_components=componentes,     # Dimension resultante de la reduccion
        learning_rate="auto",         # Euristica lr max(N / early_exaggeration / 4, 50)
        init="random",                
        perplexity=perplexity,        # Debe ser menor que el numero de muestras
        n_iter=8000,                  # El algoritmo requiere de muchas iteraciones
        n_iter_without_progress=200,  # Criterio para early stop.
        min_grad_norm=1e-7,           # Criterio para early stop.
        method="barnes_hut"           # Aproximacion numerica del metodo "exact"
    )
    
    # Reducimos la dimencionalidad de los datos
    low_dim_space = tsne_obj.fit_transform(muestras)

    # Dibujamos los clusters
    clusters_names = np.unique(clustering_map)   # Cantidad de clusters totales
    labels_names   = np.unique(labels_map)       # Cantidad de labels totales
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for cluster in clusters_names:
        muestras_del_cluster = low_dim_space[clustering_map == cluster]
        x_data = muestras_del_cluster[:, 0]
        y_data = muestras_del_cluster[:, 1]
        ax[0].scatter(x_data, y_data, label=f"Cluster {cluster}")
        ax[0].legend()

    for label in labels_names:
        muestras_de_label = low_dim_space[labels_map == label]
        x_data = muestras_de_label[:, 0]
        y_data = muestras_de_label[:, 1]
        ax[1].scatter(x_data, y_data, label=f"Label {label}")
        ax[1].legend()

    fig.suptitle("Reduccion dimensional TSNE")
    plt.savefig(fname+"_tse.png")
    plt.close()


def plot_clustering_pca(muestras, clustering_map, labels_map, fname, n_components=2):
    """
    Realiza una reduccion de dimensionalidad de los datos proporcionados, 
    usando el metodo Principal Component Analisis (PCA) y dibuja el 
    resultado de esta reduccion y lo guarda en un archivo
    Args:
        muestras        :  Array de vectores a reducir dimensionalidad
        clustering_map  :  Mapeo de las muestras a cada cluster
        labels_map      :  Mapeo de las muestras a cada label.
        fname           :  Nombre de la imagen resultante
        n_components    :  Dimension resultante. Default [2]
    """
    # Reductor PCA
    pca_obj = PCA(n_components=n_components)
    
    # Reducimos la dimensionalidad de los datos
    low_dim_space = pca_obj.fit_transform(muestras)

    # Dibujamos los clusters  
    clusters_names = np.unique(clustering_map)   # Cantidad de clusters totales
    labels_names   = np.unique(labels_map)       # Cantidad de labels totales
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    
    for cluster in clusters_names:
        muestras_del_cluster = low_dim_space[clustering_map == cluster]
        x_data = muestras_del_cluster[:, 0]
        y_data = muestras_del_cluster[:, 1]
        ax[0].scatter(x_data, y_data, label=f"Cluster {cluster}")
        ax[0].legend()

    for label in labels_names:
        muestras_de_label = low_dim_space[labels_map == label]
        x_data = muestras_de_label[:, 0]
        y_data = muestras_de_label[:, 1]
        ax[1].scatter(x_data, y_data, label=f"Label {label}")
        ax[1].legend()
    
    fig.suptitle("Reduccion dimensional PCA")
    plt.savefig(fname+"_pca.png")
    plt.close()


def plot_clustering_tsne_pca(muestras, fname, clustering_map, labels_map, 
                             c_tsne=2, perplexity=50, c_pca=10):
    """
    Reduce la dimensionalidad de los datos con pca y luego a ese resultado
    le aplica tsne y guarda los resultados en una imagen.

    Con el fin de acelerar el calculo y reducir el ruido que ocurre 
    en el algoritmo tsne en espacios de muchas dimensiones, se recomienda
    reducir la dimensionalidad con otro algoritmo ademas de tsne. 
    Como los words embedding son densos es apropiado usar PCA

    Args:
        muestras         :  Array de vectores a reducir dimensionalidad
        fname            :  Nombre de la imagen resultante
        clustering_map   :  Mapeo de las muestras a cada cluster
        labels_map       :  Mapeo de las muestras a cada label.
        c_tsne           :  Dimension resultante luego de tsne. Default [2]
        perplexity       :  Parametro relacionado a algoritmos de manifold learning.
                            Default [50].
        c_pca            :  Dimension resultante luego de PCA. Default [10]
    """
    # Reductor TSNE
    tsne_obj = TSNE(
        n_components=c_tsne,          # Dimension resultante de la reduccion
        learning_rate="auto",         # Euristica lr max(N / early_exaggeration / 4, 50)
        init="random",                
        perplexity=perplexity,        # Debe ser menor que el numero de muestras
        n_iter=8000,                  # El algoritmo requiere de muchas iteraciones
        n_iter_without_progress=200,  # Criterio para early stop.
        min_grad_norm=1e-7,           # Criterio para early stop.
        method="barnes_hut"           # Aproximacion numerica del metodo "exact"
    )

    # Reductor PCA
    pca_obj = PCA(n_components=c_pca)

    # Reduccion intermedia
    espacio_intermedio = pca_obj.fit_transform(muestras)

    # Reduccion final
    espacio_final = tsne_obj.fit_transform(espacio_intermedio)

    # Dibujamos los clusters
    clusters_names = np.unique(clustering_map)   # Cantidad de clusters totales
    labels_names   = np.unique(labels_map)       # Cantidad de labels totales
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for cluster in clusters_names:
        muestras_del_cluster = espacio_final[clustering_map == cluster]
        x_data = muestras_del_cluster[:, 0]
        y_data = muestras_del_cluster[:, 1]
        ax[0].scatter(x_data, y_data, label=f"Cluster {cluster}")
        ax[0].legend()
    
    for label in labels_names:
        muestras_de_label = espacio_final[labels_map == label]
        x_data = muestras_de_label[:, 0]
        y_data = muestras_de_label[:, 1]
        ax[1].scatter(x_data, y_data, label=f"Label {label}")
        ax[1].legend()

    fig.suptitle("Reduccion dimensional PCA + TSNE")
    plt.savefig(fname+"_pca_tsne.png")
    plt.close()


def array_to_tsv(vectores, fname):
    """
    Convierte un array en un archivo tsv, que se puede utilizar en 
    TensorFlow Projector. Cada vector se convierte en una fila del documento tsv
    Args:
        vectores    :    Array de arrays.
        fname       :    Nombre del archivo resultante.
    """
    with open(fname, 'w') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        for vector in vectores:
            tsv_output.writerow(vector)


# Funciones para manipular datos (Mumuki)
# =============================================================================
def get_enunciado(item_id, items):
    """
    Devuelve el enunciado y el nombre asociado al problema con identificador 'item_id'
    Args:
        item_id : Identificador del problema objetivo
        items   : Filas asociadas a 'item_id'
    Return:
        enunciado : Enunciado y nombre del problema asociado a item_id
    """
    try:
        name, description = items[items["id"] == item_id]\
                                   [["name", "description"]].values.flatten()
        return f"[{name} - {item_id}] :: {description}"
    except ValueError:
        return f"La id {item_id} no corresponde a un ejercicio valido del dataset"