import json
import math
import pandas as pd
import time
import argparse
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import f1_score

# 带入自定义的执行参数
parser = argparse.ArgumentParser(description='训练情绪分类语言模型')
parser.add_argument('--data', required=True, help='训练数据文件路径')
args = parser.parse_args()


def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')  # 可選 'micro', 'macro', 'weighted'

def flatten_data(nested_data):
    """Flatten the nested list of lists into a single list of dictionaries."""
    flat_data = []
    for sublist in nested_data:
        for item in sublist:
            flat_data.append({"text": item[0], "labels": int(item[1])})
    return flat_data

def getDataFrame(data_path):
    try:
        with open(data_path, 'r', encoding="utf8") as file:
            nested_data = json.load(file)
        
        # 平展数据
        flat_data = flatten_data(nested_data)
        
        # 转换为 DataFrame
        df = pd.DataFrame(flat_data)
        
        return df
    except Exception as e:
        print(f"Error reading data file: {e}")
        raise

def train(df):
    # 输出语言模型的目录名称
    dir_name = 'bert-base-Chinese-bs-64-epo-3'

    # 自定义参数
    model_args = ClassificationArgs()
    model_args.train_batch_size = 64
    model_args.num_train_epochs = 3  # 確認使用3個epoch
    model_args.output_dir = f"outputs/{dir_name}"
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 1000  # 每1000步進行驗證

    # 切分數據為訓練集和驗證集
    train_df = df.sample(frac=0.8, random_state=42)  # 80% 用於訓練
    eval_df = df.drop(train_df.index)  # 20% 用於驗證

    # 建立ClassificationModel
    model = ClassificationModel(
        'bert', 
        'bert-base-chinese', 
        use_cuda=True, 
        num_labels=6,  # 確保這裡的 num_labels 與你的分類數量一致
        args=model_args
    )
    
    # 訓練模型
    model.train_model(train_df, eval_df=eval_df, f1=f1_multiclass)  # 傳入自定義的 f1 函數

    '''
    # 在每個周期結束後保存結果
    eval_results_output_dir = os.path.join(os.getcwd(), 'SimpleTransformerTest')
    os.makedirs(eval_results_output_dir, exist_ok=True)

    for epoch in range(model_args.num_train_epochs):
        print(f"Evaluating after epoch {epoch+1}...")
        results, _, _ = model.eval_model(eval_df, f1=f1_multiclass)  # 傳入 f1 函數
        
        # 寫入文件
        with open(f"{eval_results_output_dir}/validation_results_epoch_{epoch+1}.txt", "w") as f:
            f.write(f"Validation results after epoch {epoch+1}:\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
    '''


# 主程序
if __name__ == "__main__":
    tStart = time.time()  # 計時開始
    df = getDataFrame(args.data)
    train(df)
    tEnd = time.time()  # 計時結束

    # 輸出程序執行的時間
    print(f"執行花費 {tEnd - tStart} 秒")