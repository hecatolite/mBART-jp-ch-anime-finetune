import whisper
import os
import json
import transformers
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
    TrainerCallback,
)
def change_time_format(time):
    """
    将时间格式从秒转换为 HH:MM:SS 格式。

    :param time: 时间，单位为秒
    :return: 转换后的时间字符串
    """
    time = round(time, 2)
    #print(time)
    hours = math.floor(time / 3600)
    minutes = math.floor((time % 3600) / 60)
    seconds = math.floor(time % 60)
    milliseconds = round(time-int(time), 2)
    milliseconds = int(milliseconds * 100)
    if milliseconds >=100:
        milliseconds /= 10
        milliseconds = int(milliseconds)

    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:02d}"
def convert_to_ass(json_data, ch_font_name="SimHei", ch_font_size=16, jp_font_name="DFMaruGothic Std W7", jp_font_size=12):
    """
    将 JSON 数据转换为 ASS 格式的字幕，并实现 ch 和 jp 字幕同步显示，jp 显示在下方且字体较小。

    :param json_data: JSON 格式的字幕数据
    :param ch_font_name: ch 字幕的字体名称
    :param ch_font_size: ch 字幕的字体大小
    :param jp_font_name: jp 字幕的字体名称
    :param jp_font_size: jp 字幕的字体大小
    :return: ASS 格式的字幕字符串
    """
    # ASS 文件头
    ass_content = "[Script Info]\n"
    ass_content += "Title: Subtitles\n"
    ass_content += "ScriptType: v4.00+\n"
    ass_content += "WrapStyle: 0\n"
    ass_content += "ScaledBorderAndShadow: yes\n"
    ass_content += "PlayResX: 384\n"
    ass_content += "PlayResY: 288\n\n"

    # 定义样式
    ass_content += "[V4+ Styles]\n"
    ass_content += "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
    # CHS 样式
    ass_content += f"Style: CH,{ch_font_name},{ch_font_size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,1,2,10,10,10,128\n"
    
    # JAP 样式
    ass_content += f"Style: JP,{jp_font_name},{jp_font_size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,1,2,10,10,10,128\n\n"

    # 字幕事件
    ass_content += "[Events]\n"
    ass_content += "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"

    # 遍历 JSON 数据，生成字幕
    for entry in json_data:
        ch = entry["ch"]
        jp = entry["jp"]
        start = entry["start"].replace(",", ".")  # ASS 格式使用点作为毫秒分隔符
        end = entry["end"].replace(",", ".")      # ASS 格式使用点作为毫秒分隔符

        # 添加 ch 字幕行
        ass_content += f"Dialogue: 0,{start},{end},CH,,0,0,0,,{ch}\n"
        # 添加 jp 字幕行
        ass_content += f"Dialogue: 0,{start},{end},JP,,0,0,0,,{jp}\n"

    return ass_content

def save_ass_file(ass_content, output_file="subtitles.ass"):
    """
    将 ASS 内容保存到文件。

    :param ass_content: ASS 格式的字幕内容
    :param output_file: 输出文件名
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(ass_content)
def load_model_and_tokenizer(model_path, model_name="facebook/mbart-large-50-many-to-many-mmt"):
    if model_path == "base":
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        model = MBartForConditionalGeneration.from_pretrained(model_name)
    else:
        config = transformers.AutoConfig.from_pretrained(model_path)
        tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
        model = MBartForConditionalGeneration.from_pretrained(model_path, config=config)
    return model, tokenizer
text_to_trans = []
# 加载 Whisper 模型
model = whisper.load_model("turbo", device="cuda")

# 直接处理视频文件
video_path = "bocchi.mkv"
video_name = os.path.splitext(os.path.basename(video_path))[0]
# 将视频中的音频转换为文本
result = model.transcribe(video_path, language='Japanese', word_timestamps=False)

# 保存结果到 JSON 文件
output_path = "transcription_result.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

for segment in result['segments']:
    # 文本及对应时间轴
    text_to_trans.append({"text": segment['text'], "start": segment['start'], "end": segment['end']})


model1, tokenizer1 = load_model_and_tokenizer("./mbart-ja2zh-model")
translated_texts = []

for item in text_to_trans:
    inputs = tokenizer1(item['text'], return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model1.generate(**inputs)
    translated_text = tokenizer1.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    translated_texts.append({
        "ch": translated_text,
        "jp": item['text'],
        "start": change_time_format(item['start']),
        "end": change_time_format(item['end'])
    })

# 保存翻译结果到 JSON 文件
translated_output_path = "translated_result.json"
with open(translated_output_path, "w", encoding="utf-8") as f:
    json.dump(translated_texts, f, ensure_ascii=False, indent=4)

# 将翻译结果转换为subtitle
subtitle_content = convert_to_ass(translated_texts)
save_ass_file(subtitle_content, video_name+".ass")
