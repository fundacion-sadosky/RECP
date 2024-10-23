import csv
from sys import argv

discs_path = argv[1]
msg_path = argv[2]
items_path = argv[3]

with open(items_path, newline='', encoding='utf8') as items:
    items_reader = csv.DictReader(items)
    languages = {}
    for item in items_reader:
        languages[item['id']] = item['language_id']

with open(discs_path, newline='',encoding='utf8') as discs:
    discs_reader = csv.DictReader(discs)
    newfile = discs_path[:-4] + '_js.csv'
    with open(newfile, 'w', newline='', encoding='utf8') as discsJS:
        discsJS_writer = csv.DictWriter(discsJS, fieldnames=discs_reader.fieldnames)
        discsJS_writer.writeheader()
        jsIDs = set()
        for disc in discs_reader:
            if languages[disc['item_id']] == '5':
                discsJS_writer.writerow(disc)
                jsIDs.add(disc['id'])

with open(msg_path, newline='',encoding='utf8') as msgs:
    msgs_reader = csv.DictReader(msgs)
    newfile = msg_path[:-4] + '_js.csv'
    with open(newfile, 'w', newline='', encoding='utf8') as msgsJS:
        msgsJS_writer = csv.DictWriter(msgsJS, fieldnames=msgs_reader.fieldnames)
        msgsJS_writer.writeheader()
        for msg in msgs_reader:
            if msg['discussion_id'] in jsIDs:
                msgsJS_writer.writerow(msg)