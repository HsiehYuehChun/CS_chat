# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:24:28 2024

@author: rdf_series
"""
import os
import sys
import torch
import torch.nn as nn 
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import time
from typing import Dict
import socket
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from sqlalchemy import create_engine

print("Running bot_multi_sensitive")
bot_name = 'bot_multi_sensitive'

try:    
    mother_folder = sys.argv[1] ## 線上主機中的檔案位置
except:    
    mother_folder = r"C:\Users\user\Downloads\文字審核\測案2-4語系4標籤" ## 線上主機中的檔案位置，線上，到客戶資料夾為止，EX:客戶資料夾名稱= test1
sys.path.append(f"{mother_folder}/{bot_name}")  ## 指到自己的Folder
import setting_config
from transformers import Trainer, TrainingArguments
import json               #提供處理JSON格式資料的功能
app = FastAPI()
print("API: Ready to GO")
import logging

logger = logging.getLogger(__name__)
# Define Pydantic model for prediction response
class PredictResponse(BaseModel):
    predicted_class: str
    probabilities: Dict[str, float]
    inference_time: float

from typing import List, Dict

class Result(BaseModel):
    name: str
    accept: bool
    score: float

class PredictResponse(BaseModel):
    predicted_class: List[Result]
    inference_time: float

## ip
host_ip = setting_config.ip
port_ip = setting_config.port
# Predefined labels and variables
labels = ['porn', 'offensive', 'politics', 'crime']  # Update labels

label_mapping = {
    'porn': 'porn',
    'offensive': 'offensive',
    'politics': 'politics',
    'crime': 'crime',
    'advertisement': 'advertisement'
}
model = None
tokenizer = None
thresholds = {}
# Initialize model and tokenizer at startup
@app.on_event("startup")
async def startup():
    global model, tokenizer, thresholds
    model_path = 'xlm-roberta-base'  # XLM-RoBERTa base model
    save_path = os.path.join(mother_folder, bot_name, "Models")  # Correct local path

    tokenizer, model = load_model_and_tokenizer(model_path, save_path)

    if model is None or tokenizer is None:
        print("Error: Model or tokenizer failed to load!")
        raise HTTPException(status_code=500, detail="Model or tokenizer failed to load.")
    else:
        print("Model and tokenizer loaded successfully.")

    # 在啟動時讀取門檻值
    thresholds = get_threshold_from_db()
    print("Thresholds loaded successfully at startup.")

@app.get("/")
def hello():
    return 'Hello, I am Bot multi sensitive words filter'

def get_threshold_from_db():
    try:
        env = setting_config.env
        server = setting_config.DB_info['address']
        username = setting_config.DB_info['uid']
        password = setting_config.DB_info['pwd']
        data = json.load(open(f'{mother_folder}/{bot_name}/config.json'))[env]
        threshold_list = data["Threshold"]
        database = threshold_list.split('.')[0].strip('[]')

        engine = create_engine(f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server')
        df_thresholds = pd.read_sql(f"SELECT * FROM {data['Threshold']}", engine)

        required_keys = ['SimilarityTextDetectPorn', 'SimilarityTextDetectOffensive', 'SimilarityTextDetectPolitics', 'SimilarityTextDetectCrime']

        for required_key in required_keys:
            if required_key not in df_thresholds['Key'].values:
                raise HTTPException(status_code=500, detail=f"Threshold for {required_key} not found in the database.")

        df_thresholds = df_thresholds[df_thresholds['Key'].isin(required_keys)]
        df_thresholds['Key'] = pd.Categorical(df_thresholds['Key'], categories=required_keys, ordered=True)
        df_thresholds = df_thresholds.sort_values('Key')

        df_thresholds['Value'] = df_thresholds['Value'].apply(pd.to_numeric, errors='coerce')

        if df_thresholds['Value'].isnull().any():
            raise HTTPException(status_code=500, detail="Some threshold values could not be converted to float.")

        threshold_dict = {}
        for index, row in df_thresholds.iterrows():
            try:
                label = row['Key'].replace('SimilarityTextDetect', '').lower()
                threshold_dict[label] = row['Value']
            except Exception as e:
                print(f"Error processing key: {row['Key']}, error: {e}")
                continue
        return threshold_dict

    except ValueError as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"讀取資料庫時發生未知錯誤：{str(e)}")
        raise HTTPException(status_code=500, detail=f"讀取資料庫時發生未知錯誤：{str(e)}")
        
# 新增更新門檻值的端點
@app.get("/updatefaq")
async def update_thresholds():
    global thresholds
    thresholds = get_threshold_from_db()
    print("Thresholds updated successfully.")
    return {"message": "Thresholds updated successfully."}

# Prediction endpoint
@app.get("/predict", response_model=PredictResponse)
async def predict(text: str):
    start_time = time.time()
    try:
        if not text:
            raise HTTPException(status_code=400, detail="Please provide the text.")

        if model is None or tokenizer is None:
            raise HTTPException(status_code=500, detail="Model or tokenizer is not loaded properly.")
        print(f"Received text: {text}")

        match_values = {}
        csv_path = os.path.join(mother_folder, bot_name, 'multi_filter.csv')
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=500, detail=f"CSV file not found at {csv_path}")

        sensitivity_data = pd.read_csv(csv_path)
        matching_rows = sensitivity_data[sensitivity_data['Text'].apply(lambda x: text.find(x) != -1)]

        if not matching_rows.empty:
            for index, row in matching_rows.iterrows():
                for label in ['Porn', 'Offensive', 'Politics', 'Crime']:
                    if row.get(label) is not None and row.get(label) > 0:
                        match_values[label.lower()] = row[label]
        else:
            print("No matching rows found in multi_filter.csv")

        results, inference_time = predict_text(text, model, tokenizer, labels)

        # 使用全局變數中的門檻值
        global thresholds

        for result in results:
            label = result['name']
            if label == 'advertisement':
                continue  # 'advertisement' 的 accept 已直接設定，無需門檻值比較
    
            mapped_label = label_mapping.get(label)
    
            if mapped_label:
                threshold = thresholds.get(mapped_label, 0.9)
                if threshold is None:
                    threshold = 0.9
                print(f"Using threshold for {label}: {threshold}")
            else:
                print(f"Warning: No mapped label found for '{label}', using default threshold 0.9")
                threshold = 0.9
    
            if result.get('score') is None:
                result['score'] = 0.0
    
            accept = result['score'] >= threshold
            result['accept'] = accept

        return PredictResponse(
            predicted_class=results,
            inference_time=inference_time
        )

    except Exception as e:
        print(f"Error in /predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
# Prediction helper function
def predict_text(text, model, tokenizer, labels, max_length=40):
    try:
        start_time = time.time()
        
        # Tokenize the text
        inputs = tokenizer(text, padding='max_length', truncation=True, max_length = max_length, return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Make predictions
        outputs = model(input_ids, attention_mask)
        logits = outputs.logits
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        probs = torch.round(probs * 1000) / 1000  # Round the probabilities to 3 decimal places
        
        probs_list = probs[0].tolist()  # Convert tensor to list

        # Check if the length of probs_list matches the length of labels
        if len(probs_list) != len(labels):
            raise ValueError("probs_list and labels must have the same length.")

        global thresholds
        
        results = []
        if len(probs_list) != len(labels):
            raise ValueError("probs_list and labels must have the same length.")
        
        for i, label in enumerate(labels):
            score = f"{float(probs_list[i]):.3f}"
            threshold_str = thresholds.get(label, "0.90")
            try:
                threshold = float(threshold_str)
            except ValueError:
                print(f"Warning: Could not convert value '{threshold_str}' to float for label '{label}'. Using default value 0.90.")
                threshold = 0.90

            accept = float(score) >= threshold

            results.append({
                'name': label,
                'accept': accept,
                'score': float(score)
            })
        # 比對外部連結標籤化檔案
        external_csv_path = os.path.join(mother_folder, bot_name, 'external.csv')
        if not os.path.exists(external_csv_path):
            raise HTTPException(status_code=500, detail=f"CSV file not found at {external_csv_path}")
        
        external_data = pd.read_csv(external_csv_path)
        
        # 使用正則表達式進行精確匹配，確保只匹配完整單詞
        matching_external_rows = external_data[external_data['Text'].apply(lambda x: bool(re.search(re.escape(x), text)))]
        
        advertisement_found = False
        
        if not matching_external_rows.empty:
            advertisement_found = True
            print(f"Matching rows found at indices: {matching_external_rows.index.tolist()}")
        
        # 新增 advertisement 欄位
        results.append({
            'name': 'advertisement',
            'accept': advertisement_found,
            'score': 1.0 if advertisement_found else 0.0
        })
        
        inference_time = round(time.time() - start_time, 3)  # Round the inference time to 3 decimal places
        return results, inference_time  # Return the list of results and inference time

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
# Helper function to preprocess the text (removes stopwords)
def preprocess_text(text, stop_words_set):
    tokens = text.split()
    filtered_tokens = [token for token in tokens if token not in stop_words_set]
    return " ".join(filtered_tokens)

# CustomDataset 類別應該根據以下修改來接收 attention_mask
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data, labels):
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

# Function to retrain model with new data
@app.get("/train")
async def train_model_from_db(background_tasks: BackgroundTasks, epoch: int = 3, learning_rate: float = 0.000001):
    try:
        # Respond immediately that training has started in the background
        background_tasks.add_task(train_model, epoch, learning_rate)
        return {"message": "Training started in the background"}

    except Exception as e:
        print(f"Error in /train endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

training_stopped = False  # Flag to stop training

# Function to load model and tokenizer
def load_model_and_tokenizer(model_path, save_path):
    if not os.path.exists(save_path):
        print(f"Model directory not found: {save_path}")
        return None, None
    try:
        # Load the model for XLM-RoBERTa
        model = XLMRobertaForSequenceClassification.from_pretrained(save_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None
    try:
        # Load the tokenizer for XLM-RoBERTa
        tokenizer = XLMRobertaTokenizer.from_pretrained(save_path)
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Tokenizer not found at {save_path}, downloading from {model_path}")
        try:
            tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
            tokenizer.save_pretrained(save_path)
            print("Tokenizer downloaded and saved successfully")
        except Exception as download_error:
            print(f"Error downloading tokenizer: {str(download_error)}")
            return None, None
    model.eval()
    return tokenizer, model

# Modify the training function to check for the flag
def train_model(epoch: int, learning_rate: float):
    global training_stopped
    try:
        # SQL connection and data import setup
        env = setting_config.env
        server = setting_config.DB_info['address']
        database = ''
        username = setting_config.DB_info['uid']  
        password = setting_config.DB_info['pwd']

        try:
            # Read training data from the database
            data = json.load(open(f'{mother_folder}/{bot_name}/config.json'))[env]
            MultiLanBan_list = data["MultiLanBan"]
            database = MultiLanBan_list.split('.')[0].strip('[]')
            # 建立 SQLAlchemy engine
            engine = create_engine(f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server')

            # 使用 pandas.read_sql() 函數讀取資料 (現在使用 engine)
            train_data = pd.read_sql(f"SELECT * FROM {data['MultiLanBan']}", engine)
            train_data = add_sensitivity_columns(train_data)

        except Exception as data_error:
            print(f"Error during data import: {str(data_error)}")
            raise HTTPException(status_code=500, detail="Data import failed")

        # Proceed with the training if data import is successful
        # Split data into training and validation sets
        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

        # Process stop words
        stop_words_csv_path = os.path.join(mother_folder, bot_name, '/Models/stopword.csv')
        stop_words_df = pd.read_csv(stop_words_csv_path)
        stop_words_set = set(stop_words_df.iloc[:, 0].tolist())
        
        # Preprocess text (remove stop words)
        train_inputs = [preprocess_text(text, stop_words_set) for text in train_data["Text"].tolist()]
        val_inputs = [preprocess_text(text, stop_words_set) for text in val_data["Text"].tolist()]

        # Tokenize and create attention masks
        tokenized_train = tokenizer(train_inputs, padding='max_length', truncation=True, max_length=40, return_tensors='pt')
        tokenized_val = tokenizer(val_inputs, padding='max_length', truncation=True, max_length=40, return_tensors='pt')

        # Prepare labels
        labels = ['porn', 'offensive', 'politics', 'crime']
        train_labels = torch.tensor(train_data[labels].values).float()
        val_labels = torch.tensor(val_data[labels].values).float()

        # Create custom dataset
        train_dataset = CustomDataset(tokenized_train, train_labels)
        val_dataset = CustomDataset(tokenized_val, val_labels)

        # Set up training parameters
        training_args = TrainingArguments(
            output_dir='./results', 
            eval_strategy="epoch",
            save_strategy="epoch", 
            learning_rate=learning_rate, 
            per_device_train_batch_size=8, 
            per_device_eval_batch_size=8, 
            num_train_epochs=epoch, 
            weight_decay=0.3, 
            load_best_model_at_end=True, 
            metric_for_best_model="f1"
        )

        # Set up optimizer and loss function
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        # Create custom trainer to use optimizer and loss function
        class CustomTrainer(Trainer):
            def __init__(self, *args, optimizer=None, criterion=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.optimizer = optimizer
                self.criterion = criterion

            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss = self.criterion(logits, labels)
                return (loss, outputs) if return_outputs else loss

        # Initialize trainer and start training
        trainer = CustomTrainer(
            model=model, 
            args=training_args, 
            train_dataset=train_dataset, 
            eval_dataset=val_dataset,
            optimizer=optimizer, 
            criterion=criterion
        )

        # Start training process
        for epoch in range(epoch):
            if training_stopped:  # Check if the training should be stopped
                print("Training stopped early.")
                break
            trainer.train()

        # Save the retrained model
        trainer.save_model("retrained_model")
        tokenizer.save_pretrained("retrained_model")

        print("Training complete!")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

def add_sensitivity_columns(df):
    # 初始化四个新列，默认填充为0
    df["porn"] = 0
    df["offensive"] = 0
    df["politics"] = 0
    df["crime"] = 0
    
    # 使用 apply 函数基于 'SynoemotionName' 列的值更新相应的列
    def update_columns(row):
        if row["SynoemotionName"] == "政治敏感詞":
            row["politics"] = 1
        elif row["SynoemotionName"] == "情色敏感詞":
            row["porn"] = 1
        elif row["SynoemotionName"] == "髒字辱罵":
            row["offensive"] = 1
        elif row["SynoemotionName"] == "違法犯罪":
            row["crime"] = 1
        return row
    
    # 应用 update_columns 函数到每一行
    df = df.apply(update_columns, axis=1)
    
    # 最后只保留 'Text', '色情', '冒犯', '政治', '犯罪' 这五个列
    df = df[['Text', 'porn', 'offensive', 'politics', 'crime']]
    return df


@app.get("/stop_training")
async def stop_training():
    global training_stopped
    try:
        # Set the flag to stop training
        training_stopped = True
        return {"message": "Training process has been stopped."}
    except Exception as e:
        print(f"Error in /stop_training endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop training: {str(e)}")
        
if __name__ == "__main__":
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    import uvicorn
    uvicorn.run(app, host=host_ip, port=port_ip)
