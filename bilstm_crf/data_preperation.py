import json

from tools.sqlite_tool import select_names

for name in ['country', 'province', 'grape', 'price']:
    data_filled_list = []
    print(name + 'now')
    template = '<{}>'.format(name)
    file_name = name + '_data'
    if name is 'grape':
        name = 'variety'
    data_filling_list = select_names(name)
    if name is 'price':
        data_filling_list = [('cheap',), ('medium',), ('expensive',)]
    with open('/path/bt'+ '/data_labeled' + '/data_template/' + file_name) as f:
        for line in f:
            # ','->' '
            line = line.replace(',', ' ')
            line = line.replace('.', '')
            line = line.replace('。', '')
            line = line.replace('?', '')
            try:
                index_in = line.index('in')
                if line[index_in + 2] != ' ' and line[index_in - 1] == ' ':
                    line = line[:index_in] + 'in ' + line[index_in + 2:]
                    print(line)
            except Exception as e:
                pass
            line = ' '.join(line.split())
            for data_filling in data_filling_list:
                data_filling = data_filling[0]
                data_filled = line.replace(template, data_filling)
                data_filled_split = data_filled.split()
                tags = ['O'] * len(data_filled_split)
                try:
                    slot_index = data_filled_split.index(data_filling.split()[0])
                except Exception as e:
                    print(e)
                    continue
                tags[slot_index] = 'B-' + name
                if len(data_filling.split()) > 1:
                    for num in range(slot_index + 1, slot_index + len((data_filling.split()))):
                        tags[num] = 'I-' + name
                    # i_tag = ['I-' + name] * (len(data_filling.split()) - 1)
                    # bias = len(data_filling.split()) - 1
                    # tags[slot_index + 1:slot_index + len(data_filling.split()) - 1] = i_tag
                data_filled_list.append({'key':data_filled.strip(), 'value':data_filling, 'tags':tags})
                print(data_filled)
                tags = []

    with open('/path/bt/data_labeled/data_sequence_labeling/data_sl.json', 'a') as f:
        for ddd in data_filled_list:
            f.write(json.dumps(ddd))
            f.write('\n')