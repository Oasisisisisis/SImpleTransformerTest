from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import pandas as pd
import json

def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def predict(model, listTestData):
    predictions, raw_outputs = model.predict(listTestData)
    return predictions

if __name__ == "__main__":
    tStart = time.time()

    # 讀取數據和標籤
    data = read_data("data.json")
    labels = [int(label) for label in read_data("labels.json")]

    # 訓練模型
    model_args = ClassificationArgs()
    model_args.train_batch_size = 64
    model_args.num_train_epochs = 3
    model_args.overwrite_output_dir = True  # 設置覆蓋輸出目錄

    model = ClassificationModel(
        'bert',
        r'C:\SimpleTransformerTest\outputs\bert-base-Chinese-bs-64-epo-3',
        use_cuda=True,
        cuda_device=0,
        num_labels=6,
        args=model_args
    )

    # 格式化訓練數據
    train_data_with_labels = pd.DataFrame({
        'text': data,
        'labels': labels
    })

    # 在測試集上進行預測
    finalpredict = predict(model, data)

    # 繪製混淆矩陣
    cm = confusion_matrix(labels, finalpredict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["other", "like", "sadness", "disgust", "anger", "happiness"])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    # 输出分類報告
    report = classification_report(labels, finalpredict, target_names=["other", "like", "sadness", "disgust", "anger", "happiness"])
    print("分類报告:")
    print(report)

    tEnd = time.time()
    print(f"執行花費 {tEnd - tStart:.2f} 秒。")
