import os
import json
import random

def get_json(file_path, json_path):
    name_list = os.listdir(file_path)
    save_file = []
    for i in name_list:
        path_dir = file_path + i + '/'
        for j in os.listdir(path_dir):
            tuple_now = {}
            path_now = path_dir + j
            label = int(i)
            tuple_now['image'] = path_now
            tuple_now['label'] = label
            save_file.append(tuple_now)
    #print(save_file)
    random.shuffle(save_file)
    #print(save_file)
    with open(json_path, 'w') as f:
        json.dump(save_file, f)
    

if __name__ == '__main__':
    train_file_path = '/home/public_datasets/dzj_img_5k/train/'
    test_file_path = '/home/public_datasets/dzj_img_5k/test/'
    save_train = 'train.json'
    save_test  = 'test.json'
    get_json(train_file_path, save_train)
    get_json(test_file_path, save_test)
