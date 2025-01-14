import re
import json
from collections import defaultdict

def parse_ass_to_jsonl(ass_file, jsonl_file):
    # 读取 .ass 文件
    with open(ass_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 初始化变量
    jp_subtitles = defaultdict(list)  # 存储日文字幕，按时间戳分组
    cn_subtitles = defaultdict(list)  # 存储中文字幕，按时间戳分组
    subtitle_pairs = []  # 存储中日字幕对

    # 正则表达式匹配 Dialogue 行
    dialogue_pattern = re.compile(r"^Dialogue: \d+,(\d+:\d+:\d+\.\d+),(\d+:\d+:\d+\.\d+),.*,,(.*)$")

    # 遍历每一行
    for line in lines:
        # 匹配 Dialogue 行
        match = dialogue_pattern.match(line.strip())
        if match:
            start_time = match.group(1)  # 开始时间
            end_time = match.group(2)  # 结束时间
            text = match.group(3).strip()  # 字幕文本
            # 去除格式信息（如 {\fad(0,150)}）
            text = re.sub(r"\{.*?\}", "", text).strip()
            # 根据行类型（JP 或 CN）存储字幕
            if "JP" in line:
                jp_subtitles[(start_time, end_time)].append(text)
            elif "CN" in line:
                cn_subtitles[(start_time, end_time)].append(text)

    # 将日文字幕和中文字幕按时间戳配对
    for time_key in jp_subtitles:
        if time_key in cn_subtitles:
            # 如果时间戳匹配，将日文字幕和中文字幕配对
            for jp_text in jp_subtitles[time_key]:
                for cn_text in cn_subtitles[time_key]:
                    subtitle_pairs.append([jp_text, cn_text])

    # 将字幕对写入 jsonl 文件
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for pair in subtitle_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"字幕对已成功写入 {jsonl_file}")

# 示例调用
ass_file = "[DBD-Raws][幼女战记][01-12TV全集+SP+剧场版][1080P][BDRip][HEVC-10bit][FLAC][MKV]/[DBD-Raws][幼女战记][01][1080P][BDRip][HEVC-10bit][FLAC].ass" 
jsonl_file = "subtitles.jsonl"  # 输出的 .jsonl 文件路径
parse_ass_to_jsonl(ass_file, jsonl_file)


