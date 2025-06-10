# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 17:14:16 2025

@author: user
"""
import os
os.system('cls')
# 清除所有變量（謹慎使用） 
for name in dir(): 
    if not name.startswith('_'): 
        del globals()[name]

import os
import pandas as pd 
import torch 
import torch.nn as nn 
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, Trainer, TrainingArguments 
from sklearn.model_selection import train_test_split 
from torch.utils.data import Dataset, DataLoader 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 指定支持中文的字型，這裡使用 SimHei 作為例子（你可以根據實際情況調整）
rcParams['font.family'] = 'SimHei'  # 使 Matplotlib 支持中文
rcParams['axes.unicode_minus'] = False  # 防止顯示負號時出現亂碼

# 也可以設置 Seaborn 使用 Matplotlib 設定的字型
sns.set(font="Microsoft YaHei")  

os.chdir("C:/Users/user/Downloads/Telegram Desktop/多語敏感詞訓練")
os.makedirs("./results", exist_ok=True)
# Set directory and ensure results folder exists
base_dir = r"C:/Users/user/Downloads/Telegram Desktop/多語敏感詞訓練" 
#results_dir = os.path.join(base_dir, "/results") 
results_dir = r"C:/Users/user/Downloads/TrainingLogs" 
# 直接指定讀取絕對路徑
data_folder = r"C:\Users\user\Downloads\Telegram Desktop\敏感加停用\整理完待更新資料庫"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

for root, dirs, files in os.walk(results_dir): 
    for file in files: 
        print(os.path.join(root, file))

# Clear existing TensorBoard logs before each training
if os.path.exists(results_dir):
  shutil.rmtree(r"C:/Users/user/Downloads/TrainingLogs")  # Remove the entire directory
os.makedirs(r"C:/Users/user/Downloads/TrainingLogs")  # Recreate the directory

os.chdir(base_dir)
# Print to verify paths 
print(f"Base Directory: {base_dir}")
print(f"Results Directory: {results_dir}")

# 設定訓練設備 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"Using device: {device}")

# 初始化 TensorBoard 的 SummaryWriter\
writer = SummaryWriter(log_dir=results_dir)

# 指定要用於訓練的 CSV 列表
csv_files_to_read = ['簡中一般對話完整標籤化.csv', '簡中政治對話完整標籤化.csv', '多語系訓練資料集.csv','英文政治對話完整標籤化.csv','CSsource.csv', '推特標籤化不良對話_updated.csv','Gossiping-QA-Dataset-2_0_updated.csv']#
dfs = [pd.read_csv(os.path.join(data_folder, file)) for file in csv_files_to_read]

df6 = dfs[6]
df6 = df6.sample(n=310000)#
dfs[6] = df6

df = pd.concat(dfs, ignore_index=True, join='inner')
columns_to_convert = ['色情', '冒犯', '政治', '犯罪']
for column in columns_to_convert: 
    df[column] = df[column].astype(float)

# 重新隨機排列各列 
df = df.sample(frac=1).reset_index(drop=True)
# 處理可能的 NaN 值 
df = df.fillna("")

csv_stopword = ['停用詞大雜燴.csv']
stop_words = [pd.read_csv(os.path.join(data_folder, file)) for file in csv_stopword]
stop_words_df = stop_words[0]
stop_words_set = set(stop_words_df.iloc[:, 0].tolist())#

def preprocess_text(text): 
    if isinstance(text, str):
        # Tokenize text (optional, adjust based on your tokenizer)
        tokens = text.split()
        
        # Remove stop words
        filtered_tokens = [token for token in tokens if token not in stop_words_set]
        
        # Join filtered tokens back into a string
        preprocessed_text = " ".join(filtered_tokens)
    else:
        preprocessed_text = ""
    return preprocessed_text

inputs = df["Text"].tolist()
inputs = [preprocess_text(text) for text in inputs]
 
# 檢查組合後的部分文本 
print("Input examples:", inputs[:10])

# 預訓練的 BERT tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

class CustomXLMRobertaForSequenceClassification(XLMRobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self._init_weights(self.classifier)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask cannot be None")
    
        # 模型前向傳播
        outputs = self.roberta(input_ids, attention_mask=attention_mask, **kwargs)
    
        # 檢查 pooler_output 是否為 None，若是則使用 last_hidden_state
        if outputs.pooler_output is None:
            print("WARNING: pooler_output is None, using last_hidden_state instead.")
            pooled_output = outputs.last_hidden_state.mean(dim=1)  # 進行平均池化
        else:
            pooled_output = outputs.pooler_output
    
        # 此處應返回 [batch_size, num_labels] 形狀的 logits
        logits = self.classifier(pooled_output)  # 通過分類器得到 logits

        loss = None
        if labels is not None:
            # 確保 labels 的形狀與 logits 一致
            labels = labels.view(logits.size())
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
    
        return {"loss": loss, "logits": logits}


# 初始化 mBERT 模型，輸出維度根據任務目的調整(此任務標準為6標籤輸出)
# model = CustomXLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=4)
model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=4)


## 凍結部分 XLM-Roberta 模型的層
num_layers = model.config.num_hidden_layers
for name, param in model.roberta.named_parameters():
    num_layers = model.config.num_hidden_layers
    last_two_layers = [num_layers - 2, num_layers - 1]
    last_two_layers_names = [f'encoder.layer.{i}' for i in last_two_layers]

    if any(layer_name in name for layer_name in last_two_layers_names) or 'pooler' in name:
        param.requires_grad = True  # 解凍最後兩層和池化層
    else:
        param.requires_grad = False  # 凍結其他層

model.config.hidden_dropout_prob = 0.25
model.config.attention_probs_dropout_prob = 0.25

# 將模型移動到對應的設備 
model.to(device)

# 資料處理
# 將DataFrame 轉換為 Hugging Face Dataset 格式
tokenized_inputs = tokenizer(
    inputs, 
    padding='max_length', 
    truncation=True, 
    max_length=40, 
    return_tensors='pt'
    )

input_ids = tokenized_inputs['input_ids'].to(torch.long) 
attention_mask = tokenized_inputs['attention_mask'].to(torch.long)
print("Tokenized Input IDs:", tokenized_inputs['input_ids'][:10])
print("Attention Masks:", tokenized_inputs['attention_mask'][:10])

# 將 labels 轉換為 PyTorch Tensor
labels = torch.tensor(df[['色情', '冒犯', '政治', '犯罪']].values).to(torch.float) #, '禁用語', '外部連結'

print("Labels:", labels[:10])

# 定義訓練參數
training_args = TrainingArguments(
    output_dir=results_dir, 
    eval_strategy="epoch",
    save_strategy="epoch", 
    learning_rate=0.000009, 
    per_device_train_batch_size=30, 
    per_device_eval_batch_size=30, 
    num_train_epochs=10, 
    weight_decay=0.2, 
    load_best_model_at_end=True, 
    metric_for_best_model="f1" 
    )

# 定義 dataset 負責將 input_ids, attention_mask, labels 打包成一個字典 方便 DataLoader 讀取數據。
class CustomDataset(Dataset): 
    def __init__(self, encodings, labels): 
        self.encodings = encodings 
        self.labels = labels  # 保持一熱編碼格式，不再轉換成類別索引
    def __getitem__(self, idx): 
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()} 
        item['labels'] = self.labels[idx].clone().detach() 
        return item 
    def __len__(self): 
        return len(self.labels)

# 分割 input_ids 和 attention_mask
train_inputs, temp_inputs,train_masks, temp_masks, train_labels, temp_labels = train_test_split(input_ids,attention_mask, labels, test_size=0.2)
val_inputs, test_inputs,val_masks, test_masks, val_labels, test_labels = train_test_split(temp_inputs, temp_masks, temp_labels, test_size=0.5)

train_dataset = CustomDataset({'input_ids': train_inputs, 'attention_mask': train_masks}, train_labels)
val_dataset = CustomDataset({'input_ids': val_inputs, 'attention_mask': val_masks}, val_labels)
train_dataloader = DataLoader(train_dataset, batch_size=30, shuffle=True)
eval_dataloader = DataLoader(val_dataset, batch_size=30)

def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, vmin=0)
    plt.title(title)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.tight_layout()
    return plt.gcf() # 使用 gcf() 來獲取當前的 figure

def compute_metrics(eval_pred, threshold=0.4):
    logits, labels = eval_pred

    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits).to(device)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels).to(device)

    sigmoid = nn.Sigmoid()
    probs = sigmoid(logits).cpu().numpy()

    safe_predictions = np.all(probs < threshold, axis=1)

    predictions = np.argmax(probs, axis=1)
    labels = labels.cpu().numpy()

    # 將 one-hot encoded labels 轉換為類別索引
    labels = np.argmax(labels, axis=1)

    # 正確的處理方式
    labels_with_safe = np.where(safe_predictions, 4, labels)
    predictions_with_safe = np.where(safe_predictions, 4, predictions)

    cm = confusion_matrix(labels_with_safe, predictions_with_safe, labels=np.arange(5))
    figure = plot_confusion_matrix(cm, labels=["色情", "冒犯", "政治", "犯罪", "安全"])
    writer.add_figure("Confusion Matrix", figure, global_step=eval_pred[0].shape[0])

    accuracy = accuracy_score(labels_with_safe, predictions_with_safe)
    precision = precision_score(labels_with_safe, predictions_with_safe, average='weighted', zero_division=0)
    recall = recall_score(labels_with_safe, predictions_with_safe, average='weighted', zero_division=0)
    f1 = f1_score(labels_with_safe, predictions_with_safe, average='weighted', zero_division=0)

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }

# 定義 "安全" 類別的處理函數
def compute_safe_classification(logits, threshold=0.4):
    """
    判斷是否將某個樣本標記為 "安全" 類別，根據所有標籤的預測值是否都低於 threshold。
    """
    sigmoid = nn.Sigmoid()
    probs = sigmoid(logits)
    
    # Convert probs from a tensor to a numpy array
    probs = probs.cpu().detach().numpy()

    # 判斷預測值是否都低於 threshold
    safe_predictions = np.all(probs < threshold, axis=1).astype(int)  # 這裡使用 axis=1 來檢查每個樣本的所有類別預測
    return safe_predictions

# 定義優化器
optimizer = torch.optim.AdamW(model.parameters(), lr=0.000009) #torch.optim.AdamW

# 损失函数
criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵

# 定義 Trainer
trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=val_dataset, 
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None)
)

# 定義訓練輪數/早停参数
num_epochs = 30  # 可以根據需求調整這個值
best_val_loss = float('inf')  # 初始設為正無窮
patience = 0  # 初始化耐心次數
max_patience = 9  # 最大耐心次數
   
is_first_epochs = True  # 控制前兩個 epoch 是否重新分割數據
# 開始訓練
for epoch in range(num_epochs):
    '''
    if is_first_epochs:
        # 每个 epoch 开始时重新分割数据
        train_inputs, temp_inputs, train_masks, temp_masks, train_labels, temp_labels = train_test_split(input_ids, attention_mask, labels, test_size=0.2)
        val_inputs, test_inputs, val_masks, test_masks, val_labels, test_labels = train_test_split(temp_inputs, temp_masks, temp_labels, test_size=0.5)
        
        # 將 is_first_two_epochs 設置為 False，在第三個 epoch 開始不再重新分割
        if epoch == 0:  # 第2個epoch過後就不再重新分割
            is_first_two_epochs = False
    else:
        # 如果不是前兩個epoch，則使用之前的資料分割結果
        pass  # 不需要做任何處理，繼續使用已經分割好的資料集
    
    # 创建数据集
    train_dataset = CustomDataset({'input_ids': train_inputs, 'attention_mask': train_masks}, train_labels)
    val_dataset = CustomDataset({'input_ids': val_inputs, 'attention_mask': val_masks}, val_labels)
    
    # 創建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=30, shuffle=True)
    eval_dataloader = DataLoader(val_dataset, batch_size=30)
    '''
    model.train()  
    epoch_loss = 0
    global_step = 0
    time_start = time.time()
    
    # Training loop
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs['logits']
        labels = batch['labels']
        
        # 確保 labels 的形狀與 logits 一致
        labels = labels.view(logits.size())      
        loss = criterion(logits, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1


        # 打印損失值以進行監控
        print(f"Loss: {loss.item()}")
    
    # Evaluation at the end of each epoch (instead of after 15 epochs)
    model.eval()
    eval_metrics = trainer.evaluate()  # Evaluate the model on the validation set
    val_loss = eval_metrics['eval_loss']
    val_accuracy = eval_metrics['eval_accuracy']

    # Log metrics to TensorBoard after each epoch
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/val", val_accuracy, epoch)

    # Evaluate on the test dataset
    test_dataset = CustomDataset({'input_ids': test_inputs, 'attention_mask': test_masks}, test_labels)
    test_metrics = trainer.evaluate(test_dataset)
    test_accuracy = test_metrics['eval_accuracy']

    # Log test accuracy to TensorBoard
    writer.add_scalar("Accuracy/test", test_accuracy, epoch)

    # Save the model if it's the best validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss  # Update the best validation loss
        patience = 0  # Reset patience counter
        trainer.save_model(f"best_model_epoch_{epoch}")  # Save the best model
        tokenizer.save_pretrained(f"best_model_epoch_{epoch}")  # Save the tokenizer
        print(f"Model saved at epoch {epoch} with val_loss: {val_loss}")
    else:
        patience += 1  # Increase patience counter
        if patience >= max_patience:  # Trigger early stopping if patience is reached
            print("Early stopping triggered!")
            break  # End training early
    # Print the epoch's statistics
    print(f"Epoch: {epoch}, Step: {global_step}")  

    # Timing for the epoch
    time_end = time.time()
    time_c = time_end - time_start
    print("------------------------------------")
    print('Time cost:', time_c, 's')
    print('Time cost hours:', time_c / 3600, 'hr')
    print("                                    ")

# Close the writer after training
writer.close()

# Evaluate on the test dataset after training
tokenizer = XLMRobertaTokenizer.from_pretrained("best_model")  # Load the saved tokenizer
model = CustomXLMRobertaForSequenceClassification.from_pretrained("best_model")
model.to(device)
test_dataset = CustomDataset({'input_ids': test_inputs, 'attention_mask': test_masks}, test_labels)
test_metrics = trainer.evaluate(test_dataset)
print(test_metrics)
