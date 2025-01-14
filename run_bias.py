import json
import random
import os
import wandb
import math

import nltk
from nltk.translate.bleu_score import corpus_bleu
import torch
import random
from datasets import Dataset
from sklearn.model_selection import train_test_split
# Transformers & SentencePiece
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    EarlyStoppingCallback,
)

# 定义数据集文件路径
train_file = 'split_datasets/train.json'
val_file = 'split_datasets/val.json'

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    japanese_sentences = [pair[0] for pair in data]
    chinese_sentences = [pair[1] for pair in data]
    
    return japanese_sentences, chinese_sentences

# 加载训练集
train_japanese, train_chinese = load_data(train_file)

# 加载验证集
val_japanese, val_chinese = load_data(val_file)

# 创建 Hugging Face Dataset 对象
train_dataset = Dataset.from_dict({
    'japanese': train_japanese,
    'chinese': train_chinese
})

val_dataset = Dataset.from_dict({
    'japanese': val_japanese,
    'chinese': val_chinese
})
val_dataset = val_dataset.shuffle(seed=42).select(range(500))   # 只取其中500条，方便测试

# 查看训练集的第一个样本
print(train_dataset[0])

model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

def preprocess_function(examples):
    # 源语言：日文
    tokenizer.src_lang = "ja_XX"  

    # 同时处理 source 和 target
    model_inputs = tokenizer(
        examples['japanese'], 
        text_target=examples['chinese'],  # 直接用 text_target 来处理 labels
        max_length=128,
        truncation=True,
        padding='max_length'
    )

    return model_inputs

# 定义保存路径
train_cache_path = '/data/yangzhizhuo/NLP/tokenized_train_dataset'
val_cache_path = '/data/yangzhizhuo/NLP/tokenized_eval_dataset'

# 如果缓存文件存在，直接加载已处理的数据
if os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
    print("加载缓存的训练数据和验证数据...")
    tokenized_train = Dataset.load_from_disk(train_cache_path)
    tokenized_val = Dataset.load_from_disk(val_cache_path)
else:
    # 否则进行数据预处理并保存到硬盘
    print("开始预处理训练数据和验证数据...")
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    
    # 保存预处理后的数据
    tokenized_train.save_to_disk(train_cache_path)
    tokenized_val.save_to_disk(val_cache_path)
    print("预处理完成并保存数据到硬盘.")

#print(tokenized_train[0])

# 仅微调偏置
for name, param in model.named_parameters():
    if 'bias' not in name:
        param.requires_grad = False

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
)

print("downloading")
# 评估指标
try:
    nltk.data.find('~/nltk_data/tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
print("finish downloading")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # 将 -100 去掉后再解码
    labels = [[l for l in label if l != -100] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 计算 BLEU
    references = [[ref.split()] for ref in decoded_labels]
    hypotheses = [hyp.split() for hyp in decoded_preds]
    bleu = corpus_bleu(references, hypotheses)

    return {"bleu": bleu}

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

wandb.init(project="NLP_project", name="run4")
trainer.train(resume_from_checkpoint = "/data/yangzhizhuo/NLP/results/checkpoint-16500")
wandb.finish()

model.save_pretrained("/data/yangzhizhuo/NLP/mbart-ja2zh-model")
tokenizer.save_pretrained("/data/yangzhizhuo/NLP/mbart-ja2zh-model")
