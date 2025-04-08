import easyocr
from PIL import Image
import PIL.Image
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import re

# 修补Pillow的ANTIALIAS属性
PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

# 设置Matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def ocr_image(image_path):
    """使用EasyOCR从图像中提取文本"""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")
        reader = easyocr.Reader(['ch_sim', 'en'],
                                model_storage_directory='C:/Users/99362/.EasyOCR/model',
                                gpu=True if device == 'cuda' else False)
        result = reader.readtext(image_path)
        text = [item[1] for item in result]
        print(f"OCR结果 ({image_path}): {text}")
        return text
    except Exception as e:
        print(f"OCR处理图像 {image_path} 时出错: {e}")
        return []


def clean_item(item):
    """清理消费项目，去除无关信息"""
    # 移除时间字符串
    item = re.sub(r'\d+月\d+日\s?\d+:\d+', '', item)
    item = re.sub(r'\b(美团|收|已全额退款|收银)\b', '', item).strip()
    # 移除多余空格
    item = re.sub(r'\s+', ' ', item).strip()
    return item


def parse_text(text):
    """解析OCR文本，提取金额、时间和消费项目"""
    data_list = []
    current_item = []
    current_time = None
    current_amount = None
    i = 0

    while i < len(text):
        item = text[i]

        # 跳过非记录行（如“2025年4月”）
        if "年" in item and "月" in item and "日" not in item:
            i += 1
            continue

        # 提取时间（格式：X月X日 HH:MM 或 X月X日HH:MM）
        time_match = re.search(r'(\d+月\d+日\s?\d+:\d+)', item)
        if time_match:
            # 如果找到时间，说明是一条新记录
            if current_time and current_amount and current_item:
                # 保存上一条记录
                cleaned_item = clean_item(' '.join(current_item))
                if cleaned_item:  # 确保消费项目不为空
                    data = {
                        'amount': current_amount,
                        'time': f"2025年{current_time}",
                        'item': cleaned_item
                    }
                    print(f"提取数据: {data}")
                    data_list.append(data)

            # 重置当前记录
            current_item = []
            current_amount = None
            # 规范化时间格式，确保“日”后有空格
            time_str = time_match.group(1)
            time_str = re.sub(r'(日)(\d+:\d+)', r'\1 \2', time_str)
            current_time = time_str

            # 寻找最近的金额（向前或向后搜索）
            # 向前搜索（金额可能在时间之前）
            for j in range(i - 1, max(-1, i - 5), -1):
                if j < 0:
                    break
                amount_match = re.search(r'([+-]?\d+\.\d+)', text[j])
                if amount_match:
                    try:
                        current_amount = float(amount_match.group(1))
                        break
                    except ValueError:
                        pass

            # 向后搜索（金额可能在时间之后）
            if not current_amount:
                for j in range(i + 1, min(len(text), i + 5)):
                    amount_match = re.search(r'([+-]?\d+\.\d+)', text[j])
                    if amount_match:
                        try:
                            current_amount = float(amount_match.group(1))
                            i = j  # 跳到金额位置
                            break
                        except ValueError:
                            pass

            i += 1
            continue

        # 如果不是时间，累积到消费项目
        current_item.append(item)
        i += 1

    # 处理最后一条记录
    if current_time and current_amount and current_item:
        cleaned_item = clean_item(' '.join(current_item))
        if cleaned_item:
            data = {
                'amount': current_amount,
                'time': f"2025年{current_time}",
                'item': cleaned_item
            }
            print(f"提取数据: {data}")
            data_list.append(data)

    return data_list


def process_images(image_dir):
    """处理多个图像文件，提取数据"""
    data_list = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            text = ocr_image(image_path)
            records = parse_text(text)
            data_list.extend(records)
    return data_list


def main(image_dir):
    """主函数，处理数据并生成图表"""
    data_list = process_images(image_dir)
    if not data_list:
        print("未找到有效数据，请检查截图或调整关键词。")
        return

    df = pd.DataFrame(data_list)
    # 规范化时间格式，确保一致
    df['time'] = df['time'].str.replace(r'(日)(\d+:\d+)', r'\1 \2', regex=True)
    df['time'] = pd.to_datetime(df['time'], format='%Y年%m月%d日 %H:%M', errors='coerce')
    df = df.dropna(subset=['time'])  # 移除无法解析的时间
    df['date'] = df['time'].dt.date
    daily_spending = df.groupby('date')['amount'].sum()

    plt.figure(figsize=(10, 6))
    daily_spending.plot(kind='line')
    plt.title('每日消费金额总结')
    plt.xlabel('日期')
    plt.ylabel('总金额 (CNY)')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    image_dir = 'C:/Users/99362/Desktop/work/test'
    main(image_dir)