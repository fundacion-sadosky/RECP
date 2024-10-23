from functools import reduce
from datetime import datetime
import spacy as sp
from random import shuffle
import re
from string import punctuation
from emoji import demojize, replace_emoji

stop_chars_def = ['\n\r', '\n', '\r', '\t', '\v']
dt_format_def = '%Y-%m-%d %H:%M:%S.%f'

## Funcion para crear las conversaciones (con mensajes ordenados por fecha)
## Se construye un diccionario en donde se clasifica por ejercicio (por defecto).
## Se puede opcionalmente guardar los ids de las conversaciones, limitar por mensajes aprobados o sólo mensajes de tutores (en cuyo caso hacen falta sus uids).
## Por último, agregamos una opción para guardar información de los mensajes (en cuyo caso quedan separados, pero en orden).
def build_chats(discs, msgs, format=dt_format_def, use_ids=False, use_exs=True, approved_msgs=False, students_msgs=False, approved_ids=[], msg_info=False, reject_frags=[]):
    all_chats = {} if use_ids or use_exs else []
    for disc in discs:
        responses = []
        for msg in msgs:
            if disc['id'] == msg['discussion_id']:
                if students_msgs or (approved_msgs and msg['approved'] == 'True') or int(msg['sender_uid']) in approved_ids:
                    if not any(rej in msg['content'] for rej in reject_frags):
                        responses.append(msg)
        
        responses.sort(key=lambda msg: datetime.strptime(msg['created_at'], format))
        if msg_info:
            textInfo = [{'ID': int(msg['id']), 'sender': int(msg['sender_uid']), 'text': msg['content']} for msg in responses]
            if students_msgs:
                textInfo.insert(0, {'ID': int(disc['id']), 'sender': int(disc['initiator_uid']), 'text': disc['description']})
        else:
            textInfo = reduce(lambda text, res: text + ' ' + res['content'], responses, disc['description'] if students_msgs else '')
        if len(textInfo) > 0:
            if use_exs:
                if use_ids:
                    all_chats[disc['item_id']] = {**(all_chats.get(disc['item_id'], {})), **{disc['id']: textInfo}}
                else:
                    all_chats[disc['item_id']] = all_chats.get(disc['item_id'], []) + [textInfo]
            else:
                if use_ids:
                    all_chats[disc['id']] = textInfo
                else:
                    all_chats.append(textInfo)
    
    return all_chats


## Funcion para contar todas las conversaciones (clasificadas en un diccionario)
def total_chats(chats):
    def count_chats(count, ex):
        if isinstance(ex, dict):
            return count + total_chats(ex)
        else:
            return count + (len(ex) if isinstance(ex, list) else 1)
    return reduce(count_chats, list(chats.values()), 0)


## Funcion para filtrar chats individualmente
def filter_chats(chats, filter_fun):
    filtered_chats = {}
    for ex, ex_chats in chats.items():
        if isinstance(ex_chats, dict):
            filtered = filter_chats(ex_chats, filter_fun)
            if filtered:
                filtered_chats[ex] = filtered
        elif isinstance(ex_chats, list):
            filtered = list(filter(filter_fun, ex_chats))
            if filtered:
                filtered_chats[ex] = filtered
        else:
            if filter_fun(ex_chats):
                filtered_chats[ex] = ex_chats
    
    return filtered_chats


## Funcion para filtrar clases (ejercicios) globalmente
def filter_classes(chats, filter_fun):
    return {ex: chats for ex, chats in chats.items() if (isinstance(chats, list) and filter_fun(chats)) or (isinstance(chats, dict) and filter_fun(list(chats.values())))}


## Funcion para separar los chats en train, test y (opcionalmente) dev
## Solo para chats ordenados por ejercicio!
def split_chats(all_chats, train_percent, dev_percent=0):
    if train_percent + dev_percent > 100:
        print('Los porcentajes suman más del total!')
        return 0
    else:
        train_chats, test_chats, dev_chats = {}, {}, {}
        for ex, chats in all_chats.items():
            slice_i = round(train_percent*len(chats)/100)
            slice_j = round((train_percent+dev_percent)*len(chats)/100)
            shuffle(chats)
            train_chats[ex] = chats[:slice_i]
            dev_chats[ex] = chats[slice_i:slice_j]
            test_chats[ex] = chats[slice_j:]
        return (train_chats, test_chats, dev_chats)


## Funcion auxiliar para reorganizar el diccionario, separando las conversaciones de las clases correspondientes
## Es importante que preserve el orden!
def add_class(chats, item):
    chats[0] = chats[0] + item[1] if isinstance(item[1], list) else chats[0] + [item[1]]
    chats[1] = chats[1] + [item[0]]*(len(item[1]) if isinstance(item[1], list) else 1)
    return chats

# Función que reorganiza el diccionario.
def separate_classes(chats):
    return reduce(add_class, list(chats.items()), [[], []])


# Función específica para "limpiar" el texto de acuerdo con diversos criterios.
# En general vamos a preferir reemplarar los caracteres especiales (como '\n') por un espacio (en lugar de quitarlos).
def clean_text(text, lowercase=False, remove_spaces=True, remove_punctuation=False, remove_emojis=False, rem_rep_chars=False):
    if lowercase:
        text = text.lower()
    if remove_spaces:
        text = " ".join(text.split())
    if remove_punctuation:
        text = re.sub(f"[{re.escape(punctuation)}]", "", text)
    if remove_emojis:
        text = replace_emoji(text, replace='')
    else:
        text = demojize(text, language='es')
    if rem_rep_chars:
        text = re.sub(r'(.)\1{3,}',r'\1', text)
    return text


# Función que escribe en el archivo especificado los chats (en el formato provisto por split_chats) de manera que los interprete Fasttext.
# Opcionalmente (pero lo usaremos!) aplicamos una función que procese los chats, y otra para limpiar el texto.
# Damos la opción de pasar a minúsculas.
def normalize_chats(chats, file, process=None, clean=None, filter=None):
    with open(file, 'w', encoding='utf-8') as f:
        for i, chat in enumerate(chats[0]):
            if process is not None:
                if clean is not None:
                    if filter is not None:
                        chat = " ".join(filter(process(clean(chat))))
                    else:
                        chat = " ".join(process(clean(chat)))
                else:
                    if filter is not None:
                        chat = " ".join(filter(process(chat)))
                    else:
                        chat = " ".join(process(chat))
            else:
                if clean is not None:
                    chat = clean(chat)

            f.write(f'__label__{chats[1][i]} {chat}\n')


nlp = sp.load('es_core_news_sm')
## Funcion tokenizadora alternativa
## Aprovechamos a limpiar el texto.
def tokenizer(text):
    tokens = nlp(clean_text(text))
    return [token.text for token in tokens]


## Función que construye el vocabulario de una lista de textos (con Spacy)
## Se puede devolver un conjunto o un diccionario (el cual además tiene las frecuencias de las palabras)
def vocabulary_builder(texts, count=False, preprocess=None):
    if count:
        vocab = {'total': 0, 'total_vocab': 0, 'words': {}}
        for text in texts:
            words = [token.text for token in nlp(preprocess(text))] if preprocess is not None else [token.text for token in nlp(text)]
            vocab['total'] = vocab['total'] + len(words)
            for word in words:
                current = vocab['words'].get(word, 0)
                if not current:
                    vocab['total_vocab'] = vocab['total_vocab'] + 1
                vocab['words'][word] = current + 1
        return vocab
    else:
        vocab = set()
        for text in texts:
            words = [token.text for token in nlp(preprocess(text))] if preprocess is not None else [token.text for token in nlp(text)]
            for word in words:
                vocab.add(word)
        return vocab


## Función que construye el vocabulario a partir de un diccionario de listas de palabras (ya está tokenizado) 
## 'countLocal' y 'countGlobal' para contar las ocurrencias locales (a cada ej.) o globales respectivamente.
## Adicionalmente countGlobal == False significa que el vocabulario se ordena por ejercicio (no por palabra)
def vocab_from_lists(words, countLocal=True, countGlobal=False):
    all_words = {}
    if countGlobal:
        for ex, chats in words.items():
            for chat, kws in chats.items():
                for kw in kws:
                    if not kw in all_words.keys():
                        all_words[kw] = {}
                        all_words[kw]['exs'] = {}
                    
                    if not ex in all_words[kw]['exs'].keys():
                        all_words[kw]['exs'][ex] = {}
                        all_words[kw]['exs'][ex]['chats'] = []
                        all_words[kw]['count'] = all_words[kw].get('count', 0) + 1
                    
                    all_words[kw]['exs'][ex]['chats'].append(chat)
                    if countLocal:
                        all_words[kw]['exs'][ex]['count'] = all_words[kw]['exs'][ex].get('count', 0) + 1
    else:
        for ex, chats in words.items():
            all_words[ex] = {}
            all_words[ex]['words'] = {}
            all_words[ex]['count'] = 0
            for chat, kws in chats.items():
                for kw in kws:
                    if not kw in all_words[ex]['words'].keys():
                        all_words[ex]['words'][kw] = {}
                        all_words[ex]['words'][kw]['chats'] = []
                        all_words[ex]['count'] = all_words[ex].get('count', 0) + 1
                    
                    all_words[ex]['words'][kw]['chats'].append(chat)
                    if countLocal:
                        all_words[ex]['words'][kw]['count'] = all_words[ex]['words'][kw].get('count', 0) + 1
    
    return all_words