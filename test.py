import os
import json
from datasets import Dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, AutoModelForCausalLM, AutoTokenizer
import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import nltk
import jieba
from openai import OpenAI

try:
    nltk.data.find('~/nltk_data/tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

client = OpenAI(api_key="your api key", base_url="https://api.deepseek.com")

def get_model_response(prompt: str) -> str:
    """
    Sends the prompt to the DeepSeek API and returns the response content.
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error during API call: {str(e)}"

# 定义数据加载函数
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        data = json.load(f)
    japanese_sentences = [pair[0] for pair in data]
    chinese_sentences = [pair[1] for pair in data]
    return japanese_sentences, chinese_sentences

# 定义模型加载函数
def load_model_and_tokenizer(model_path, model_name="facebook/mbart-large-50-many-to-many-mmt"):
    if model_path == "base":
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        model = MBartForConditionalGeneration.from_pretrained(model_name)
    else:
        tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
        model = MBartForConditionalGeneration.from_pretrained(model_path)
    return model, tokenizer

def load_qwen_model(model_name="Qwen/Qwen2.5-7B-Instruct"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def translate_base(text, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.eval()

    # 1. 日文翻译成英文
    tokenizer.src_lang = "ja_XX"
    tokenizer.tgt_lang = "en_XX"  # 先翻译成英文
    encoded_ja = tokenizer(text, return_tensors="pt", max_length=128, truncation=True).to(device)
    with torch.no_grad():
        generated_tokens_ja_to_en = model.generate(
            **encoded_ja,
            forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],  # 强制生成英文
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
    translated_en = tokenizer.batch_decode(generated_tokens_ja_to_en, skip_special_tokens=True)[0]
    
    # 2. 英文翻译成中文
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = "zh_CN"  # 然后翻译成中文
    encoded_en = tokenizer(translated_en, return_tensors="pt", max_length=128, truncation=True).to(device)
    with torch.no_grad():
        generated_tokens_en_to_zh = model.generate(
            **encoded_en,
            forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"],  # 强制生成中文
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
    translated_zh = tokenizer.batch_decode(generated_tokens_en_to_zh, skip_special_tokens=True)[0]

    return translated_zh

# 定义翻译函数
def translate(text, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.eval()
    tokenizer.src_lang = "ja_XX"
    encoded = tokenizer(text, return_tensors="pt", max_length=128, truncation=True).to(device)
    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"], # 指定生成中文文本
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translation

def translate_deepseek(text: str, model_response_func) -> str:
    fewshot = "把下面日文到中文的翻译补全，在 [] 中输入中文翻译结果:\n\n" +\
        "ううん　今来たところ\n翻译: [没有 我也是刚来]\n\n"
    prompt = fewshot + text + "\n翻译: "
    translation = model_response_func(prompt)
    translation = translation.strip()
    if "[" in translation:
        translation = translation.split("[")[1].split("]")[0]
    elif "\n" in translation:
        translation = translation.split("\n")[0] # 如果没有中文翻译，取第一行
    return translation

def translate_qwen(text: str, model, tokenizer) -> str:
    """
    Translates Japanese text to Chinese using Qwen model.
    """
    # 定义指令，类似于之前的 DeepSeek 翻译
    fewshot = "把下面日文到中文的翻译补全，在 [] 中输入中文翻译结果:\n\n" +\
        "ううん　今来たところ\n翻译: [没有 我也是刚来]\n\n" + \
        "あれ\n翻译: [奇怪]\n\n" +\
        "ご両親海外だから\n翻译: [毕竟父母在国外]\n\n"
    prompt = fewshot + text + "\n翻译: "
    
    # 创建消息列表
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 使用 Qwen 模型生成响应
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    
    # 获取生成的中文翻译
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    response = response.strip()
    if "[" in response:
        response = response.split("[")[1].split("]")[0]
    elif "\n" in response:
        response = response.split("\n")[0] # 如果没有中文翻译，取第一行
    return response

# 定义 BLEU 计算函数
def compute_bleu(references, hypotheses):
    references_tokenized = [[list(jieba.cut(ref))] for ref in references]   # 这里注意再套一层[]，因为 corpus_bleu 函数要求是多个参考答案的列表！（虽然可能只有一个）
    hypotheses_tokenized = [list(jieba.cut(hyp)) for hyp in hypotheses]
    bleu_score = corpus_bleu(references_tokenized, hypotheses_tokenized)
    return bleu_score

# 定义保存样例函数
def save_translation_samples(inputs, references, translations, file_path, num_samples=100):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("Index\tJapanese\tReference\tTranslation\n")
        for i in range(min(num_samples, len(inputs))):
            f.write(f"{i}\t{inputs[i]}\t{references[i]}\t{translations[i]}\n")

# 定义评估函数，检查是否已有翻译文件
def evaluate_model(model, tokenizer, test_dataset, translations_file='data_test/translations.json', device='cuda' if torch.cuda.is_available() else 'cpu'):
    references = test_dataset['chinese']
    inputs = test_dataset['japanese']
    
    # 如果翻译文件存在，直接加载翻译
    if os.path.exists(translations_file):
        print(f"加载已有的翻译数据：{translations_file}")
        with open(translations_file, 'r', encoding='utf-8') as f:
            translations = json.load(f)
    else:
        print("翻译数据不存在，开始翻译...")
        translations = []
        for text in tqdm(inputs, desc="翻译"):
            translation = translate(text, model, tokenizer, device)
            translations.append(translation)
        # 保存翻译数据
        with open(translations_file, 'w', encoding='utf-8') as f:
            json.dump(translations, f, ensure_ascii=False, indent=4)
    
    bleu = compute_bleu(references, translations)
    return bleu, translations

# 定义评估函数，检查是否已有翻译文件
def evaluate_base_model(model, tokenizer, test_dataset, translations_file='data_test/translations.json', device='cuda' if torch.cuda.is_available() else 'cpu'):
    references = test_dataset['chinese']
    inputs = test_dataset['japanese']
    
    # 如果翻译文件存在，直接加载翻译
    if os.path.exists(translations_file):
        print(f"加载已有的翻译数据：{translations_file}")
        with open(translations_file, 'r', encoding='utf-8') as f:
            translations = json.load(f)
    else:
        print("翻译数据不存在，开始翻译...")
        translations = []
        for text in tqdm(inputs, desc="翻译"):
            translation = translate_base(text, model, tokenizer, device)
            translations.append(translation)
        # 保存翻译数据
        with open(translations_file, 'w', encoding='utf-8') as f:
            json.dump(translations, f, ensure_ascii=False, indent=4)
    
    bleu = compute_bleu(references, translations)
    return bleu, translations

def evaluate_deepseek_model(model_response_func, test_dataset, translations_file='translations_deepseek.json'):
    """
    Evaluates the DeepSeek translation model by translating the test dataset and computing BLEU score.
    """
    references = test_dataset['chinese']
    inputs = test_dataset['japanese']

    # 如果翻译文件存在，直接加载翻译
    if os.path.exists(translations_file):
        print(f"加载已有的 DeepSeek 翻译数据：{translations_file}")
        with open(translations_file, 'r', encoding='utf-8') as f:
            translations = json.load(f)
    else:
        print("DeepSeek 翻译数据不存在，开始翻译...")
        translations = []
        for text in tqdm(inputs, desc="DeepSeek翻译"):
            translation = translate_deepseek(text, model_response_func)
            translations.append(translation)
        # 保存翻译数据
        with open(translations_file, 'w', encoding='utf-8') as f:
            json.dump(translations, f, ensure_ascii=False, indent=4)
    
    bleu = compute_bleu(references, translations)
    return bleu, translations

def evaluate_qwen_model(model, tokenizer, test_dataset, translations_file='translations_qwen.json'):
    """
    Evaluates the Qwen translation model by translating the test dataset and computing BLEU score.
    """
    references = test_dataset['chinese']
    inputs = test_dataset['japanese']

    # 如果翻译文件存在，直接加载翻译
    if os.path.exists(translations_file):
        print(f"加载已有的 Qwen 翻译数据：{translations_file}")
        with open(translations_file, 'r', encoding='utf-8') as f:
            translations = json.load(f)
    else:
        print("Qwen 翻译数据不存在，开始翻译...")
        translations = []
        for text in tqdm(inputs, desc="Qwen翻译"):
            translation = translate_qwen(text, model, tokenizer)
            translations.append(translation)
        # 保存翻译数据
        with open(translations_file, 'w', encoding='utf-8') as f:
            json.dump(translations, f, ensure_ascii=False, indent=4)
    
    bleu = compute_bleu(references, translations)
    return bleu, translations

    
if __name__ == "__main__":
    # 加载测试集
    test_file = 'split_datasets/test_sampled.json'
    test_japanese, test_chinese = load_data(test_file)
    test_dataset = Dataset.from_dict({
        'japanese': test_japanese,
        'chinese': test_chinese
    })
    print("测试集第一个样本：", test_dataset[0])

    # 加载模型
    base_model_name = "facebook/mbart-large-50-many-to-many-mmt"
    model_base, tokenizer_base = load_model_and_tokenizer("base", base_model_name)
    model1, tokenizer1 = load_model_and_tokenizer("/data/yangzhizhuo/NLP/mbart-ja2zh-model")
    print("加载 Qwen 模型...")
    qwen_model, qwen_tokenizer = load_qwen_model("Qwen/Qwen2.5-7B-Instruct")
    '''print("加载 Sakura 模型...")
    sakura_model, sakura_tokenizer = load_qwen_model("SakuraLLM/Sakura-7B-Qwen2.5-v1.0-GGUF")'''

    # 评估基础模型
    print("评估基础 mBART 模型...")
    bleu_base, translations_base = evaluate_base_model(model_base, tokenizer_base, test_dataset, 'data_test/translations_base.json')
    print(f"基础模型 BLEU 分数: {bleu_base:.4f}")  # 结果应基于已翻译或新翻译的文本

    # 评估模型1
    print("评估训练后的模型1...")
    bleu_model1, translations_model1 = evaluate_model(model1, tokenizer1, test_dataset, 'data_test/translations_model1.json')
    print(f"模型1 BLEU 分数: {bleu_model1:.4f}")

    # 评估 DeepSeek 模型
    print("评估 DeepSeek 翻译模型...")
    bleu_deepseek, translations_deepseek = evaluate_deepseek_model(get_model_response, test_dataset, 'data_test/translations_deepseek.json')
    print(f"DeepSeek 模型 BLEU 分数: {bleu_deepseek:.4f}")

    # 评估 Qwen 模型
    print("评估 Qwen 翻译模型...")
    bleu_qwen, translations_qwen = evaluate_qwen_model(qwen_model, qwen_tokenizer, test_dataset, 'data_test/translations_qwen.json')
    print(f"Qwen 模型 BLEU 分数: {bleu_qwen:.4f}")
    '''
    # 评估 Sakura 模型
    print("评估 Sakura 翻译模型...")
    bleu_sakura, translations_sakura = evaluate_qwen_model(sakura_model, sakura_tokenizer, test_dataset, 'data_test/translations_sakura.json')
    print(f"Sakura 模型 BLEU 分数: {bleu_sakura:.4f}")'''

    # 保存翻译样例
    save_translation_samples(
        test_dataset['japanese'],
        test_dataset['chinese'],
        translations_base,
        'data_test/translation_samples_base.txt',
        num_samples=100
    )
    save_translation_samples(
        test_dataset['japanese'],
        test_dataset['chinese'],
        translations_model1,
        'data_test/translation_samples_model1.txt',
        num_samples=100
    )
    save_translation_samples(
        test_dataset['japanese'],
        test_dataset['chinese'],
        translations_deepseek,
        'data_test/translation_samples_deepseek.txt',
        num_samples=100
    )
    save_translation_samples(
        test_dataset['japanese'],
        test_dataset['chinese'],
        translations_qwen,
        'data_test/translation_samples_qwen.txt',
        num_samples=100
    )
    '''save_translation_samples(
        test_dataset['japanese'],
        test_dataset['chinese'],
        translations_sakura,
        'data_test/translation_samples_sakura.txt',
        num_samples=100
    )'''