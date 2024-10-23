from messages_tagger.models import *
from django.contrib.auth.models import User
import csv
import random
from django.db.models import Max

User.objects.all().delete()
Conversation.objects.all().delete()
Evaluation.objects.all().delete()
LastTagged.objects.all().delete()
users = [
    {"name": "Francisco", "pass": "Mumuki2023Anotacion%", "group": 1, "first": 1},
    {"name": "Rocio", "pass": "Mumuki2023Anotacion%", "group": 2, "first": 1},
    {"name": "Franco", "pass": "Mumuki2023Anotacion%", "group": 2, "first": 2}
]
for user in users:
    newUser = User.objects.create_user(username=user["name"], password = user["pass"])
    newUser.save()
    counter = LastTagged.objects.create(user=newUser, group=user['group'], next_id=user["first"])
    counter.save()

with open('keywords-tutors-summ-divided.csv', newline='', encoding='utf8') as table:
    data = list(csv.DictReader(table))
    random.shuffle(data)
    id = 1
    for row in data:
        if (row['Oración aceptada 1'] and row['Oración rechazada 1']):
            newConv = Conversation(
                order_id = id,
                mumuki_id=row['ID conversación'],
                full_text=row['Conversación'],
                kw1=row['KW1'],
                kw2=row['KW2'],
                kw3=row['KW3'],
                kw4=row['KW4'],
                kw5=row['KW5'],
                ok_sent1=row['Oración aceptada 1'],
                ok_sent2=row['Oración aceptada 2'],
                ok_sent3=row['Oración aceptada 3'],
                rej_sent1=row['Oración rechazada 1'],
                rej_sent2=row['Oración rechazada 2'],
                rej_sent3=row['Oración rechazada 3']
            )
            newConv.save()
            
            id = id+1