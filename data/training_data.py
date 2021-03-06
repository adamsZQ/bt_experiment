import json


def get_training_data(file_prefix):
    data = []
    tags = []
    with open(file_prefix + '/data_labeled/data_sequence_labeling/data_sl.json') as d:
        for line in d:
            data_json = json.loads(line)
            sentence = data_json['key'].split()
            data.append(sentence)

            tag = data_json['tags']
            tags.append(tag)
    return data, tags


if __name__ == '__main__':
    predix = '/path/bt'
    data, tags = get_training_data(predix)
    print(len(data))
    print(len(tags))