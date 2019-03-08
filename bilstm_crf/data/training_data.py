import json


def get_training_data(file_prefix):
    training_data = []

    with open(file_prefix + '/data_labeled/data_sequence_labeling/data_sl.json') as d:
        for line in d:
            data_json = json.loads(line)
            sentence = data_json['key'].split()
            tag = data_json['tags']
            training_data.append((sentence, tag))
    return training_data


if __name__ == '__main__':
    predix = '/path/bt'
    training_data = (get_training_data(predix))
    print('a')
   # print(tags)