import json

def read_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data_labels = json.load(file)
    return data_labels

if __name__ == "__main__":
    # 讀取 JSON 文件
    json_file = 'test.json'
    data_labels = read_json(json_file)

    # 初始化數據和標籤列表
    data_list = []
    labels_list = []

    # 遍歷 JSON 文件內容並提取數據和標籤
    for sublist in data_labels:
        for data, label in sublist:
            data_list.append(data)
            labels_list.append(label)

    # 將數據和標籤寫入到分別的 .json 文件
    with open("data.json", "w", encoding="utf-8") as file:
        json.dump(data_list, file, indent=4, ensure_ascii=False)
        
    with open("labels.json", "w", encoding="utf-8") as file:
        json.dump(labels_list, file, indent=4, ensure_ascii=False)
    
    print("數據和標籤已分別寫入 data.json 和 labels.json 文件。")

