"""
Multiples funciones para normalizar codigo JavaScript
"""

import re
import numpy as np


# Funciones para normalizar codigo
# =============================================================================

def quitar_tildes(text):
    """
    Quita todas las tildes que ocurren en una string.
    Args:
        text   :   Texto de entrada con tildes
    """
    vocales = [("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"),
               ("Á", "A"), ("É", "E"), ("Í", "I"), ("Ó", "O"), ("Ú", "U")]
    for v in vocales:
        text = text.replace(v[0], v[1])

    return text


def quitar_comentarios(code_snippet):
    """
    Quita los comentarios de la string dada, los comentarios que reconoce
    son de las siguientes formas posibles:
        
        comentario de una linea         : '//' <text>* [\r\n]
        comentarios de multiples lineas : '/*' <text>* [\r\n]* '*/'

    Args:
        code_snippet  :  Texto de codigo
    """
    pattern = r'//.*|/\*[\s\S]*?\*/'
    comment_regex = re.compile(pattern)    
    return comment_regex.sub("", code_snippet)


def normalizar_argumentos(code):
    """
    Normaliza los nombres de los argumentos de las definciones encontradas en
    code, suponiendo que code contiene a lo sumo una definicion de funcion 
    (para asegurar la consistencia del orden de los argumentos, el cual no 
    esta garantizada si el codigo tiene mas de una definicion──por ejemplo si
    dos funciones comparten nombres de argumentos).

    Args:
        code  :   Code snippet a normalizar
        
    Example:
        code = "function(x, y, z)" -> "function(arg0, arg1, arg2)"
    """ 
    
    # Atrapa todo el texto que va en los argumentos de una definicion
    pattern = r"function\s+\w*\s*\((.*?)\)\s*"

    # Busca las coincidencias del patron en el codigo
    matches = re.findall(pattern, code) 

    normalized_code = code

    for args in matches:
        if len(args) < 1:
            # Caso en que la definicion no posee argumentos
            pass
        else:
            arg_list = args.split(',')                    # Lista de argumentos 
            arg_list = [arg.strip() for arg in arg_list]  # Strip de argumentos
            index = 0                                     # Indice para normalizar argumentos 

            # Normalizamos los argumentos
            for arg in arg_list:
                # A veces capturamos el capturamos "" como argumento
                # tambien hay ejemplos donde se usan enteros literales como
                # argumentos, los cuales no queremos normalizar
                if arg == "" or arg.isnumeric():
                    index += 1 # Para que el orden de los argumentos sea consistente
                    pass
                else:
                    new_name = 'arg'+str(index)
                    normalized_code = re.sub(r'\b' + arg + r'\b', 
                                            new_name, 
                                            normalized_code)
                    index += 1
    return normalized_code


def normalizar_variables(code):
    """
    Normaliza los nombres de las variables de las definciones encontradas en
    code, suponiendo que code en su texto define a lo sumo una funcion (de lo
    contrario no se garantiza que los nombres de las variables respeten
    el orden de definicion).
    Args:
        code  :   Code snippet a normalizar

    Example:
        code = "function(...){let x; var y}" -> "function(...){let var0, var var1}"
    """ 
    # Expresion regular que captura los nombres de las variables/constantes
    pattern = r'var\s+(\w+)|let\s+(\w+)|const\s+(\w+)'

    # Busca las coincidencias de pattern en el codigo
    matches = np.unique([x+y+z for x,y,z in re.findall(pattern, code)])

    normalized_code = code
    index = 0
    
    for match in matches: 
        new_var_name = 'var'+str(index)
        # Algunos usuarios definen variables que se llaman igual a la funcion que 
        # las contiene, para evitar reemplazar los nombres de la funcion hay que 
        # hacer un reemplazo mas estricto como el siguiente.

        # Verificamos si el nombre de la funcion se uso como nombre de variable 
        # y codificamos el nombre de la funcion
        if re.match(r'\bfunction\s*'+match+r'\s*\(', normalized_code):

            normalized_code = re.sub(r'\bfunction\s*'+match+r'\s*\(', 
                                     "FUN_CODING ", normalized_code)
            fun_decoding = f"function {match} ("

        # Reemplazamos las ocurrencias de las definciones de las variables por sus nombres genericos
        normalized_code = re.sub(r'\b' + match + r'\b', new_var_name, normalized_code)
        
        index += 1

        # Decodificamos el nombre de la funcion en caso de que este se halla usado como nombre de 
        # variable
        try:
            # Caso en que efectivamente el nombre de la funcion se uso como nombre de variable
            normalized_code = re.sub(r'\bFUN_CODING\b', fun_decoding, normalized_code)
        except:
            # Caso contrario
            pass

    return normalized_code

def normalizar_codigo(code):
    """
    Normaliza el codigo provisto normalizando sus variables y argumentos, 
    se supone que el codigo sin comentarios y sin caracteres de salto de linea
    (i.e. "\n" "\r" "\r\n"). NO presupone que el codigo contiene una unica definicion
    Args:
        code  :   Texto del code snippet  
    """
    if "function" not in code \
        and "let" not in code \
        and "var" not in code \
        and "const" not in code:
        # Las muestras que no definen funciones/variables/constantes
        # por lo general son triviales y no aportan informacion util al modelo
        return ""
    
    funciones = code.split("function") # Obtenemos las posibles multiples definciones
    result = ""

    for funcion in funciones:
        if len(funcion) < 5:
            # A veces el token que se produce en el split es simplemente un \n
            # por lo que lo omitimos
            pass
        else:
            # Luego del split la palabra "function" se pierde, la reestablecemos
            sub_code = "function " + funcion 
            # Aplicamos la normalizacion del codigo, como sub_code solo tiene una 
            # definicion, se satisface la suposicion de las funciones de normalizacion
            try: 
                result += normalizar_variables(normalizar_argumentos(sub_code))
            except Exception:
                # Hay al rededor de 50 muestras que tienen definiciones extrañas
                # donde los argumentos son llamados a funciones
                return ""
    return result


# Caracteres que deben considerarse como tokens individuales
CARACTERES_ESPECIALES = ["(", ")", ",", "{", "}", "+", "-", "*", "=",
                         ">", "<", "/", "&", "|", "!", "%", ";", "[", "]"] 


# Algunos escriben mal los comentarios (/* /*)
CARACTERES_PARA_BORRAR = ["\r\n", "\n", "\r", "/*"]

# Simbolos que deben considerarse como tokens individuales 
CARACTERES_COMPUESTOS = ["!===", 
                         ">===", "<===", "===>", "===<",    # erroneos pero ocurren 
                         ">==", "<==", "==>", "==<",        # erroneos pero ocurren
                         "===", "!==", "==", ">=", "<=", "!=", "&&", 
                         "=>", "=<",                        # erroneos pero ocurren
                          "||",  "++",  "--", "<<", ">>", "[]"]


def tokenizar_operadores(code):
    """
    Agrega espacios entre los caracteres y combinacion de caracteres
    que merecen ser considerados como un token individual e.g. + * / <=

    Args:
        code   :   Code snippet
    """
    for char in CARACTERES_PARA_BORRAR:
        code = code.replace(char, " ")
    
    # Codificacion de los caracteres compuestos
    for i in range(len(CARACTERES_COMPUESTOS)):
        code = code.replace(CARACTERES_COMPUESTOS[i], f" SYMBOL{i} ")

    # Separacion de caracteres especiales para que sean tokens individuales
    for char in CARACTERES_ESPECIALES:
        code = code.replace(char, f" {char} ")

    # Decodificacion de los caracteres compuestos
    for i in range(len(CARACTERES_COMPUESTOS)):
        code = code.replace(f"SYMBOL{i}", f" {CARACTERES_COMPUESTOS[i]} ")
    
    return code


def preprocesar_codigo(code):
    """
    Preprocesa el codigo de entrada, el resultado tiene las siguientes propiedades
        + sin tildes
        + sin comentarios
        + sin caracteres de salto de linea
        + listas normalizadas "[]"
        + operadores como tokens individuales
        + argumentos y variables normalizados
        + espacios en blanco consecutivos compactados en uno
    Args:
        code    :    code snippet
    """

    code = quitar_tildes(code)         # codigo sin tildes
    code = quitar_comentarios(code)    # codigo sin comentarios de usuario
    code = code.replace("[ ]", "[]")   # normalizacion de la lista vacia    
    code = tokenizar_operadores(code)  # operadores como tokens individuales
    code = normalizar_codigo(code)     # variables y argumentos normalizados
    code = ' '.join(code.split())      # Comprime las secuencias de espacios en blanco
    return code



# Funciones para normalizar reportes de test
# =============================================================================
def normalizar_test_results(test_results):
    """
    Normaliza los resultados provistos de la test suite (si los hubiese). El texto 
    resultante sera una string de la forma:

        TEST_RESULTS :: (<TEST_NAME_i> :: <RESULT_TEST_i>)+
        TEST_RESULTS :: NO_TESTS
    
    El segundo ejemplo muestra el caso en que no tengamos test suite

    Args:
        test_results  :  Reporte de las pruebas unitarias de la test suite.
    Returns:
        test_norm     :  Cadena de texto con el formato mencionado en la descripcion
    
    """
    # El siguiente patron captura los resultados de cada test unitario
    pattern = r'.*:title: "\s*(.*)\s*"\n.*:status: :(.*)\n'
    
    matches = re.findall(pattern, test_results) # Coincidencias encontradas

    if matches == []:
        # Caso en que no tenemos un reporte de test disponible
        return " TEST_RESULTS :: NO_TESTS "
    else:
        test_norm = " TEST_RESULTS :: "
        for match in matches:
            result = f"{match[0]} :: {match[1]} "
            test_norm += result

        return test_norm

def normalizar_expectation_results(expectation_results):
    """
    Normaliza los resultados provistos por el motor de evaluacion de expectativas
    (si los hubiese). El texto resultante sera una string de la forma:

        EXPECTATION_RESULTS :: (<EXPECTATION_NAME_i> :: <EXPECTATION_RESULT_i>)+
        EXPECTATION_RESULTS :: NO_EXPECTATIONS
    
    El segundo ejemplo muestra el caso en que no tengamos un reporte del motor
    de evaluacion de expectativas

    Args:
        expectation_results  :  Reporte del motor de evaluacion de expectativas.
    Returns:
        expectations_norm    :  Cadena de texto con el formato mencionado en la descripcion
    
    """
    # El siguiente patron captura los resultados de cada evaluacion
    pattern = r'.*:inspection: \s*(.*)\s*\n.*:result: :(.*)\n'
    
    matches = re.findall(pattern, expectation_results) # Coincidencias encontradas

    if matches == []:
        # Caso en que no tenemos un reporte de test disponible
        return " EXPECTATION_RESULTS :: NO_EXPECTATIONS "
    else:
        expectations_norm = " EXPECTATION_RESULTS :: "
        for match in matches:
            result = f"{match[0]} :: {match[1]} "
            expectations_norm += result
            
        return expectations_norm
    


def combinar_y_normalizar(data):
    """
    Realiza la normalizacion de codigo y reportes sobre data
    y duelve la concatenacion. La entrada debe ser una lista/tupla
    con el siguiente formato:

        <code_snippet, test_result, expectation_results>
    Args:
        data  :  Terna <code_snippet, test_result, expectation_results>
    Returns:
        Concatenacion de la normalizacion del contenido de data
    """
    code_norm = preprocesar_codigo(data[0])
    test_norm = normalizar_test_results(data[1])
    expc_norm = normalizar_expectation_results(data[2])

    return code_norm + test_norm + expc_norm