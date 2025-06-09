# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:16:45 2025

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
from transformers import XLMRobertaForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, XLMRobertaTokenizer
from sklearn.model_selection import train_test_split 
from torch.utils.data import Dataset, DataLoader 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(r"C:\Users\user\Downloads\Telegram Desktop\intent_new")
os.makedirs("./results", exist_ok=True)
# Set directory and ensure results folder exists
base_dir = r"C:\Users\user\Downloads\Telegram Desktop\intent_new" 
#results_dir = os.path.join(base_dir, "/results") 
results_dir = r"C:/Users/user/Downloads/TrainingLogs" 
# 直接指定讀取檔案的絕對路徑
data_folder = r"C:\Users\user\Downloads\Telegram Desktop\意圖資料庫"
file_path = data_folder + r"\多語系意圖訓練資料集4.csv"  # 拼接路徑
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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

df = pd.read_csv(file_path) 
#df = df.sample(n=1000)#
columns_to_convert = ['FitId']
for column in columns_to_convert: 
    df[column] = df[column].astype(float)
unique_fit_ids = sorted(df['FitId'].unique()) # Sort FitId values
fit_id_mapping = {old: new for new, old in enumerate(unique_fit_ids)} # Create a mapping to 0-20
df['FitId_mapped'] = df['FitId'].map(fit_id_mapping)
num_labels = len(fit_id_mapping)
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

# 加载預訓練的 BERT tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
#tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
tokenizer.save_pretrained("trytrysee")
model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=num_labels)

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

# 將DataFrame 轉換為 Hugging Face Dataset 格式
tokenized_inputs = tokenizer(
    inputs, 
    padding='max_length', 
    truncation=True, 
    max_length=50, 
    return_tensors='pt'
    )

input_ids = tokenized_inputs['input_ids'].to(torch.long) 
attention_mask = tokenized_inputs['attention_mask'].to(torch.long)
print("Tokenized Input IDs:", tokenized_inputs['input_ids'][:10])
print("Attention Masks:", tokenized_inputs['attention_mask'][:10])

# 將 labels 轉換為 PyTorch Tensor
labels = torch.tensor(df['FitId_mapped'].values).long().to(device)

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

# 定義 dataset 負責將 input_ids, attention_mask, labels 打包成一個字典，方便 DataLoader 讀取數據。
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

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits).to(device)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels).to(device)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    predictions = np.argmax(probs, axis=-1)
    labels = labels.cpu().numpy()

    # Remap the predictions and labels back to original FitId values
    inverse_fit_id_mapping = {v: k for k, v in fit_id_mapping.items()}
    original_labels = [inverse_fit_id_mapping[label] for label in labels]
    original_predictions = [inverse_fit_id_mapping[pred] for pred in predictions]
    
    cm = confusion_matrix(original_labels, original_predictions)
    figure = plot_confusion_matrix(cm, labels=np.arange(1, num_labels))  # FitId range from 1 to 36
    writer.add_figure("Confusion Matrix", figure, global_step=eval_pred[0].shape[0])

    accuracy = accuracy_score(original_labels, original_predictions)
    precision = precision_score(original_labels, original_predictions, average='weighted', zero_division=0)
    recall = recall_score(original_labels, original_predictions, average='weighted', zero_division=0)
    f1 = f1_score(original_labels, original_predictions, average='weighted', zero_division=0)
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }

def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                vmin=0, annot_kws={"fontsize": 8})
    plt.title(title)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return plt.gcf()

# 定義優化器
optimizer = torch.optim.AdamW(model.parameters(), lr=0.000009) 

# 損失函數
criterion = nn.CrossEntropyLoss()

# 定義 Trainer
trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=val_dataset, 
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None)
)

num_epochs = 50  # 可以根據需求調整這個值
best_val_loss = float('inf')  # 初始設為正無窮
patience = 0  # 初始化耐心次數
max_patience = 10  # 最大耐心次數
# 開始訓練
for epoch in range(num_epochs):
    model.train()  
    epoch_loss = 0
    global_step = 0
    time_start = time.time()

    # In your training loop:
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs['logits']
        labels = batch['labels']
    
        # 1. Reshape labels (if necessary) to be 1D:
        labels = labels.view(-1)  # Important!  Make sure labels are 1D.
    
        # 2. Convert labels to Long (CrossEntropyLoss requires Long tensors):
        labels = labels.long()   # Important!  Labels must be Long (integer) type.
    
        # 3. Use CrossEntropyLoss:
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        print(f"Loss: {loss.item()}")

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

    # Evaluation at the end of each epoch (instead of after 15 epochs)
    model.eval()
    eval_metrics = trainer.evaluate()  # Evaluate the model on the validation set
    val_loss = eval_metrics.get('eval_loss', float('inf'))
    val_accuracy = eval_metrics.get('eval_accuracy', 0)

    # Log metrics to TensorBoard after each epoch
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/val", val_accuracy, epoch)

    # Evaluate on the test dataset
    test_dataset = CustomDataset({'input_ids': test_inputs, 'attention_mask': test_masks}, test_labels)
    test_metrics = trainer.evaluate(test_dataset)
    test_accuracy = test_metrics.get('eval_accuracy', 0)

    # Log test accuracy to TensorBoard
    writer.add_scalar("Accuracy/test", test_accuracy, epoch)

    # Save the model if it's the best validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss  # Update the best validation loss
        patience = 0  # Reset patience counter
        #trainer.save_model(f"best_model_epoch_{epoch}")  # Save the best model
        #model.save_pretrained(f"best_model_epoch_{epoch}")
        # 使用 torch.save 儲存 model.state_dict 和 optimizer.state_dict
        '''
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.cpu().state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, os.path.join(results_dir, f"best_model_epoch_{epoch}.pth"))
        model.to(device) #將模型放回裝置上
        '''
        trainer.save_model(f"best_model_epoch_{epoch}")
        tokenizer.save_pretrained(f"best_model_epoch_{epoch}")  # Save the tokenizer
        #torch.save(model.state_dict(), os.path.join(f"best_model_epoch_{epoch}", "pytorch_model.bin"))
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
tokenizer = tokenizer.save_pretrained("best_model")  # Load the saved tokenizer
model = XLMRobertaForSequenceClassification.from_pretrained("best_model")
model.to(device)
test_dataset = CustomDataset({'input_ids': test_inputs, 'attention_mask': test_masks}, test_labels)
test_metrics = trainer.evaluate(test_dataset)
print(test_metrics)
