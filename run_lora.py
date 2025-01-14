import json
import os
import wandb
import nltk
from nltk.translate.bleu_score import corpus_bleu
import torch
from datasets import Dataset
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

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

# 定义 BLEU 计算函数
def compute_bleu(references, hypotheses):
    print("References Sample:", references[:2])
    print("===")
    print("Hypotheses Sample:", hypotheses[:2])
    references_tokenized = [[ref.split()] for ref in references]
    hypotheses_tokenized = [hyp.split() for hyp in hypotheses]
    print("Tokenized References Sample:", references_tokenized[:2])
    print("===")
    print("Tokenized Hypotheses Sample:", hypotheses_tokenized[:2])
    bleu_score = corpus_bleu(references_tokenized, hypotheses_tokenized)
    return bleu_score

# 评估指标
try:
    nltk.data.find('~/nltk_data/tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
print("完成 NLTK 数据下载")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # 将 -100 去掉后再解码
    labels = [[l for l in label if l != -100] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 计算 BLEU
    references = [[ref] for ref in decoded_labels]
    hypotheses = [hyp for hyp in decoded_preds]
    bleu = compute_bleu(references, hypotheses)

    return {"bleu": bleu}

# 配置 PEFT（使用 LoRA）
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,  # LoRA 的秩，越大模型的容量越高，但计算成本也越高
    lora_alpha=32,
    lora_dropout=0.1,
    #target_modules=["encoder.layers.0.self_attn", "decoder.layers.0.self_attn"]  # 根据模型结构选择
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]  # 直接指定 Linear 层
)

# 将模型转换为 PEFT 模型
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # 打印当前可训练的参数

# 定义训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=32,  # 根据 GPU 显存调整
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    logging_steps=100,
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
)

# 定义数据整理器
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 初始化 Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 初始化 Weights & Biases
wandb.init(project="NLP_project", name="PEFT_mBART_run")

# 开始训练
trainer.train()

# 结束 Weights & Biases
wandb.finish()

# 保存 PEFT 模型
model.save_pretrained("/data/yangzhizhuo/NLP/mbart-ja2zh-peft-model")
tokenizer.save_pretrained("/data/yangzhizhuo/NLP/mbart-ja2zh-peft-model")
