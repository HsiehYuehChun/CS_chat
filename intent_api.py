# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 14:37:46 2025

@author: user
"""

from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
import pandas as pd
import os
import sys
import torch
import torch as nn
import time
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import logging
logger = logging.getLogger(__name__)
# 設定日誌記錄
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import json
print("Running new_intent")
bot_name = 'bot_pure_intention'
if __name__ == "__main__":
    try:
        mother_folder = sys.argv[1]  # 線上主機檔案位置
    except IndexError:
        mother_folder = r"C:\Users\user\Downloads\Telegram Desktop\intent_new\test4"  # 本機測試路徑
    sys.path.append(f"{mother_folder}/{bot_name}")
# FastAPI setup
app = FastAPI()
print("API: Ready to GO")
# Predefined global variables
model = None
tokenizer = None
temperature = 1.2  # 初始溫度，當 temperature > 1 時，機率分布會更平滑
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data folder (adjust if needed)
data_folder = r"C:\Users\user\Downloads\Telegram Desktop\意圖資料庫"



import setting_config
host_ip = setting_config.ip
port_ip = setting_config.port

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
        tokenizer = AutoTokenizer.from_pretrained(save_path)
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Tokenizer not found at {save_path}, downloading from {model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.save_pretrained(save_path)
            print("Tokenizer downloaded and saved successfully")
        except Exception as download_error:
            print(f"Error downloading tokenizer: {str(download_error)}")
            return None, None
    model.eval()
    return tokenizer, model


# Stop words loading (same as in training script)
def preprocess_text(text, stop_words_set):
    if isinstance(text, str):
        tokens = text.split()
        filtered_tokens = [token for token in tokens if token not in stop_words_set]
        preprocessed_text = " ".join(filtered_tokens)
    else:
        preprocessed_text = ""
    return preprocessed_text

def predict(text):
    global tokenizer, model, temperature  # 使用全局變數 temperature

    if tokenizer is None:
        print("Error: Tokenizer is not loaded properly.")
        return None, None, None

    start_time = time.time()
    stopword_file = os.path.join(mother_folder, bot_name,'stopword.csv')
    stop_words = pd.read_csv(stopword_file)
    # 將停用詞轉換為 set，以提高查詢效率
    stop_words_set = set(stop_words['0'].tolist())  # 假設停用詞在 'word' 欄位
    preprocessed_text = preprocess_text(text, stop_words_set)
    inputs = [preprocessed_text]  # Wrap in a list for tokenizer

    tokenized_inputs = tokenizer(
        inputs,
        padding='max_length',
        truncation=True,
        max_length=50,  # Match your training max_length
        return_tensors='pt'
    )

    input_ids = tokenized_inputs['input_ids']
    attention_mask = tokenized_inputs['attention_mask']

    with torch.no_grad():  # Important for inference
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits / temperature  # 應用 Temperature Scaling

    probabilities = torch.softmax(logits, dim=-1).numpy()[0]  # Get probabilities
    predicted_label = np.argmax(probabilities)  # Get the label with highest probability
    inference_time = round(time.time() - start_time, 3)

    # Convert probabilities to percentages with 4 decimal places
    probabilities_percentage = [round(prob * 100, 4) for prob in probabilities]  # Convert each to percentage with 4 decimal places

    # Get the top 4 predictions (indices and corresponding probabilities)
    top_indices = np.argsort(probabilities)[-4:][::-1]  # Get indices of the top 4 classes (sorted in descending order)
    top_probabilities = [probabilities_percentage[i] for i in top_indices]  # Get the top 4 probabilities
    top_labels = top_indices  # These are the indices of the top 4 classes

    return predicted_label, probabilities_percentage, top_labels, top_probabilities, inference_time  # Return the top 4 labels and their probabilities

def create_intent_mapping(csv_file_path):
    """Creates a mapping dictionary from CSV file."""
    try:
        csv_file_path = os.path.join(mother_folder, bot_name,'compare_table.csv')
        df = pd.read_csv(csv_file_path, index_col='FitId')  # 設定 FitId 為索引
        index_to_fitid = {index: fit_id for index, fit_id in enumerate(df.index)}  # 建立索引到 FitId 的映射
        mapping = {}

        # 遍歷每一列 (代表語言代碼，例如 'en', 'ja', 'zh-TW')
        for language_code in df.columns:
            # 遍歷每一行 (每個 FitId)
            for fit_id, intention_name in df[language_code].items():
                if pd.notna(intention_name):  # 排除 NaN 值
                    # 將字串 "None" 轉換為 Python 的 None
                    if str(intention_name).lower() == 'none':
                        intention_name = None

                    if fit_id not in mapping:
                        mapping[fit_id] = {}
                    mapping[fit_id][language_code] = intention_name  # 為每個 FitId 設置語言對應的 IntentionName

        return mapping, index_to_fitid  # 返回 mapping 和 index_to_fitid
    except FileNotFoundError:
        print(f"CSV file not found: {csv_file_path}")
        return None, None
    except Exception as e:
        print(f"Error creating intent mapping: {e}")
        return None, None

# Load model and tokenizer on app startup
@app.on_event("startup")
async def startup():
    model_path = 'xlm-roberta-base'
    save_path = os.path.join(mother_folder, bot_name, "Models")
    global tokenizer, model
    tokenizer, model = load_model_and_tokenizer(model_path, save_path)
    if tokenizer is None or model is None:
        raise HTTPException(status_code=500, detail="Failed to load model or tokenizer.")

@app.get("/set_temperature")
async def set_temperature(temp: float = Query(..., description="The new temperature value (e.g., 1.0, 1.5, 0.8).")):
    global temperature
    temperature = temp
    return {"message": f"Temperature set to: {temperature}"}

@app.get("/predict")
async def predict_intent(
    text: str = Query(..., description="The input text to predict the intent of."),
    lan: str = Query(..., description="The language code for the prediction (e.g., 'zh-TW', 'en').")
):
    # Check if model and tokenizer are loaded properly
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded yet.")

    # Create intent mapping
    intent_mapping, index_to_fitid = create_intent_mapping("compare_table.csv")  # Replace with your CSV file path
    if intent_mapping is None:
        raise HTTPException(status_code=500, detail="Intent mapping creation failed.")

    # Predict the intent of the user input
    predicted_label, probabilities, top_labels, top_probabilities, inference_time = predict(text)

    results = []
    for i in range(4):
        fit_id = index_to_fitid.get(top_labels[i])

        if fit_id is not None:
            # Get predicted intent
            predicted_intent = intent_mapping.get(fit_id, {}).get(lan, "None")
            results.append({
                "class": int(top_labels[i]),  # 強制將 numpy.int64 轉換為 Python 的 int 類型
                "fit_id": fit_id,
                "predicted_intent": predicted_intent,
                "probability": float(top_probabilities[i])  # 也可以強制轉換為 float
            })


    return {
        "inference_time": inference_time,
        "results": results
    }
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
            try:
                with open(f'{mother_folder}/{bot_name}/config.json') as f:
                    data = json.load(f)[env]
            except Exception as e:
                print(f"Error loading config file: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to load config")
            Intent_list = data["Intent"]
            database = Intent_list.split('.')[0].strip('[]')
            # 建立 SQLAlchemy engine
            engine = create_engine(f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server')

            # 使用 pandas.read_sql() 函數讀取資料 (現在使用 engine)
            train_data = pd.read_sql(f"SELECT * FROM {data['Intent']}", engine)

        except Exception as data_error:
            print(f"Error during data import: {str(data_error)}")
            raise HTTPException(status_code=500, detail="Data import failed")
            
        try:
            stopword_file = os.path.join(mother_folder, bot_name,'stopword.csv')
            stop_words = pd.read_csv(stopword_file)
            # 將停用詞轉換為 set，以提高查詢效率
            stop_words_set = set(stop_words['0'].tolist())  # 假設停用詞在 'word' 欄位
            logging.info("Stopwords loaded successfully")
            # Proceed with the training if data import is successful
            # Split data into training and validation sets
            train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
            # Preprocess text (remove stop words)
            train_inputs = [preprocess_text(text, stop_words_set) for text in train_data["Text"].tolist()]
            val_inputs = [preprocess_text(text, stop_words_set) for text in val_data["Text"].tolist()]
        except Exception as e:
            logging.error(f"Error loading stopwords: {str(e)}")

        # Tokenize and create attention masks
        tokenized_train = tokenizer(train_inputs, padding='max_length', truncation=True, max_length=40, return_tensors='pt')
        tokenized_val = tokenizer(val_inputs, padding='max_length', truncation=True, max_length=40, return_tensors='pt')

        # Prepare labels
        train_labels = torch.tensor(train_data['FitId_mapped'].values).long().to(device)
        val_labels = torch.tensor(val_data['FitId_mapped'].values).long().to(device)


        # Create custom dataset
        train_dataset = CustomDataset(tokenized_train, train_labels)
        val_dataset = CustomDataset(tokenized_val, val_labels)

        # Set up training parameters
        training_args = TrainingArguments(
            output_dir='./results', 
            eval_strategy="epoch",
            save_strategy="epoch", 
            learning_rate=0.000001, 
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
    import uvicorn
    uvicorn.run(app, host=host_ip, port=port_ip)
