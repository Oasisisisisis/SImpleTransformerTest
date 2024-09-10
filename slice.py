import json
import random
import math

def split_json(input_file, train_file, test_file, split_ratio=0.7):
    try:
        # 讀取 JSON 文件
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # 打亂數據順序
        random.shuffle(data)
        
        # 照比例分割
        split_index = math.ceil(len(data) * split_ratio)
        
        # 切分數據
        part1 = data[:split_index]
        part2 = data[split_index:]
        
        with open(train_file, 'w', encoding='utf-8') as file:
            json.dump(part1, file, indent=4, ensure_ascii=False)
        
        with open(test_file, 'w', encoding='utf-8') as file:
            json.dump(part2, file, indent=4, ensure_ascii=False)

        print(f"Data successfully split into {train_file} and {test_file}")
    
    except FileNotFoundError:
        print(f"File {input_file} not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file {input_file}.")
    except Exception as e:
        print(f"An error occurred: {e}")

input_file = 'stc3_cecg_2017_and_2019_170w.json'
train_file = 'train.json'
test_file = 'test.json'

split_json(input_file, train_file, test_file)

