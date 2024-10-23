import fasttext as ft

def model_train(in_train, out_model, in_test, lr=1, epoch=50, wordNgrams=1, ws=5, lrUpdateRate=100, dim=100, out_test=None):
    model = ft.train_supervised(input=in_train, lr=lr, epoch=epoch, wordNgrams=wordNgrams, ws=ws, lrUpdateRate=lrUpdateRate, dim=dim)
    print('Modelo entrenado')
    
    model.save_model(out_model)
    print('\nModelo guardado')

    print('\nResultados')
    print(model.test(in_test))

    if out_test is not None:
        with open(in_test, 'r', encoding='utf-8') as test:
            with open(out_test, 'w', encoding='utf-8') as preds:
                for line in test:
                    i = line.find(' ')
                    label = line[:i].replace('__label__', '')
                    chat = line[i+1:].replace('\n', '')

                    preds.write(f'{label} {model.predict(chat)[0][0].replace("__label__", "")}\n')
        print('\nPredicciones guardadas')


def param_search(in_train, out_model, in_test, in_val, time):
    model = ft.train_supervised(input=in_train, autotuneValidationFile=in_val, autotuneDuration=time)
    print('Modelo entrenado')
    
    model.save_model(out_model)
    print('\nModelo guardado')

    print('\nResultados')
    print(model.test(in_test))