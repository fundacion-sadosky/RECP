import csv
from sys import argv

discs_path = argv[1]
msg_path = argv[2]
items_path = argv[3]
users_path = argv[4]

with open(discs_path, newline='', encoding='utf8') as discs:
    with open(msg_path, newline='', encoding='utf8') as msgs:
        with open(items_path, newline='', encoding='utf8') as items:
            with open(users_path, newline='', encoding='utf8') as users:
                discs_reader = csv.DictReader(discs)
                msgs_reader = csv.DictReader(msgs)
                items_reader = csv.DictReader(items)
                users_reader = csv.DictReader(users)

                items_ids, users_ids = set(), set()
                for item in items_reader:
                    items_ids.add(item['id'])
                for user in users_reader:
                    users_ids.add(user['uid'])

                discs_ids = set()
                items_miss, users_miss = [], []
                for disc in discs_reader:
                    if disc['item_id'] not in items_ids:
                        items_miss.append(int(disc['item_id']))
                    if disc['initiator_uid'] not in users_ids:
                        users_miss.append(int(disc['initiator_uid']))
                    discs_ids.add(disc['id'])
                if len(items_miss) > 0 or len(users_miss) > 0:
                    print(f'IDs de items faltantes: {items_miss}')
                    print(f'IDs de participantes faltantes: {users_miss}')
                else:
                    print('Tabla de discusiones consistente.')

                discs_miss, users_miss = [], []
                for msg in msgs_reader:
                    if msg['discussion_id'] not in discs_ids:
                        discs_miss.append(int(msg['discussion_id']))
                    if msg['sender_uid'] not in users_ids:
                        users_miss.append(int(msg['sender_uid']))
                if len(discs_miss) > 0 or len(users_miss) > 0:
                    print(f'IDs de discusiones faltantes: {discs_miss}')
                    print(f'IDs de participantes faltantes: {users_miss}')
                else:
                    print('Tabla de mensajes consistente.')