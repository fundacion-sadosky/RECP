import csv
from sys import argv

discs_path = argv[1]
msg_path = argv[2]

tutors = [0, 2, 34, 50, 115, 959]

with open(msg_path, newline='',encoding='utf8') as msgs:
    msgs_reader = csv.DictReader(msgs)
    approved_discs = set()
    for msg in msgs_reader:
        if msg['approved'] == 'True' or int(msg['sender_uid']) in tutors:
            approved_discs.add(msg['discussion_id'])

with open(discs_path, newline='',encoding='utf8') as discs:
    discs_reader = csv.DictReader(discs)
    newfile_T = discs_path[:-4] + '_T.csv'
    newfile_F = discs_path[:-4] + '_F.csv'
    with open(newfile_T, 'w', newline='', encoding='utf8') as useful:
        with open(newfile_F, 'w', newline='', encoding='utf8') as notUseful:
            discsT_writer = csv.DictWriter(useful, fieldnames=discs_reader.fieldnames)
            discsF_writer = csv.DictWriter(notUseful, fieldnames=discs_reader.fieldnames)
            discsT_writer.writeheader()
            discsF_writer.writeheader()
            for disc in discs_reader:
                if disc['status'] != 'closed' or disc['id'] in approved_discs:
                    discsT_writer.writerow(disc)
                else:
                    discsF_writer.writerow(disc)