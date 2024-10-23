import csv

# filtra en un nuevo archivo .csv
def hardFilter(table_path, predicate, tag):
    with open(table_path, newline='',encoding='utf8') as table:
        table_read = csv.DictReader(table)
        newfile = table_path[:-4] + tag + '.csv'
        with open(newfile, 'w', newline='', encoding='utf8') as newTable:
            table_write = csv.DictWriter(newTable, fieldnames=table_read.fieldnames)
            table_write.writeheader()
            for row in table_read:
                if predicate(row):
                    table_write.writerow(row)

# filtra en un arreglo
def softFilter(table_path, predicate):
    with open(table_path, newline='',encoding='utf8') as table:
        table_read = csv.DictReader(table)
        result = []
        for row in table_read:
            if predicate(row):
                result.append(row)
    return result

# filtra en un arreglo (o en un set los campos saveField), opcionalmente escribiendo un archivo .csv
def filter(table_path, predicate, write=False, tag='-new', saveField=''):
    with open(table_path, newline='',encoding='utf8') as table:
        table_read = csv.DictReader(table)
        if not write and len(saveField) > 0:
            result = set()
            for row in table_read:
                if predicate(row):
                    result.add(row[saveField])
            return result
        else:
            result = []
            for row in table_read:
                if predicate(row):
                    result.append(row)
            if write:
                newfile = table_path[:-4] + tag + '.csv'
                with open(newfile, 'w', newline='', encoding='utf8') as newTable:
                    table_write = csv.DictWriter(newTable, fieldnames=table_read.fieldnames)
                    table_write.writeheader()
                    table_write.writerows(result)
            if len(saveField) == 0:
                return result
            else:
                return {row[saveField] for row in result}