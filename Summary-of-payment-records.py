import easyocr
import cv2
import PIL.Image
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import ImageTk, Image, ImageEnhance  # 导入 ImageResampling
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re
import os
import torch
from datetime import datetime
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
CROP_Y, CROP_H = 300, 1700

# 配置全局参数
CONFIG = {
    "preprocess": {
        "resize_factor": 2,
        "contrast_factor": 1.5,
        "sharpness_factor": 2.0
    },
    "ocr": {
        "languages": ["ch_sim", "en"],
        "model_path": 'C:/Users/99362/.EasyOCR/model',
        "gpu_threshold": 0.4  # 低于此值的GPU显存不启用CUDA
    }
}


class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("智能消费记录分析系统")
        self.root.geometry("1200x800")

        # 初始化变量
        self.image_dir = ""
        self.current_image = None
        self.processed_images = []
        self.df = pd.DataFrame()

        # 创建界面组件
        self.create_widgets()

        # 初始化OCR阅读器
        self.reader = self.init_ocr_reader()

    def create_widgets(self):
        """创建GUI组件"""
        control_frame = ttk.LabelFrame(self.root, text="控制面板", padding=(10, 5))
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Button(control_frame, text="选择图片目录", command=self.select_directory).pack(pady=5)
        ttk.Button(control_frame, text="处理图片", command=self.process_images).pack(pady=5)
        ttk.Button(control_frame, text="导出数据", command=self.export_data).pack(pady=5)

        # 新增：放缩略图的 frame
        self.thumb_frame = ttk.Frame(control_frame)
        self.thumb_frame.pack(fill=tk.X, pady=10)

        # 原来的预览大图标签
        self.img_label = ttk.Label(control_frame)
        self.img_label.pack(pady=10)

        # 结果显示
        result_frame = ttk.LabelFrame(self.root, text="分析结果", padding=(10, 5))
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 表格显示
        self.tree = ttk.Treeview(result_frame, columns=("time", "amount", "item"), show="headings")
        self.tree.heading("time", text="时间")
        self.tree.heading("amount", text="金额")
        self.tree.heading("item", text="消费项目")
        self.tree.pack(fill=tk.BOTH, expand=True)

        # 图表显示
        self.figure = plt.Figure(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=result_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def init_ocr_reader(self):
        """初始化OCR阅读器"""
        device = 'cuda' if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= \
                           CONFIG["ocr"]["gpu_threshold"] * 1e9 else 'cpu'
        print(f"使用设备: {device}")
        return easyocr.Reader(
            CONFIG["ocr"]["languages"],
            gpu=device == 'cuda',
            model_storage_directory=CONFIG["ocr"]["model_path"],
            quantize=False  # 禁用量化提升准确率
        )

    def preprocess_image(self, image_path):
        """图像预处理"""
        img = Image.open(image_path)
        # ① 先裁剪列表区域
        w, h = img.size
        crop_top = CROP_Y
        crop_bot = min(h, CROP_Y + CROP_H)
        img = img.crop((0, crop_top, w, crop_bot))
        # 调整尺寸
        if CONFIG["preprocess"]["resize_factor"] != 1:
            new_size = (int(img.width * CONFIG["preprocess"]["resize_factor"]),
                        int(img.height * CONFIG["preprocess"]["resize_factor"]))
            img = img.resize(new_size, Image.LANCZOS)  # 使用 ImageResampling.LANCZOS

        # 增强对比度
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(CONFIG["preprocess"]["contrast_factor"])

        # 锐化处理
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(CONFIG["preprocess"]["sharpness_factor"])

        # 转换为OpenCV格式进行进一步处理
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return Image.fromarray(thresh)

    def ocr_image(self, path):
        proc = self.preprocess_image(path)
        raw = self.reader.readtext(np.array(proc), paragraph=False, detail=1)
        raw = sorted(raw, key=lambda x: x[0][0][1])
        texts = [t[1].strip() for t in raw]
        # 合并日期和时间，并统一时间格式
        out = []
        i = 0
        date_re = re.compile(r'^\d{1,2}月\d{1,2}日$')
        time_re = re.compile(r'^\d{1,2}[:\.]\d{2}$')  # 匹配 HH:MM 或 HH.MM
        while i < len(texts):
            if date_re.match(texts[i]) and i + 1 < len(texts) and time_re.match(texts[i + 1]):
                time_str = texts[i + 1].replace('.', ':')  # 将点号替换为冒号
                out.append(f"{texts[i]} {time_str}")
                i += 2
            else:
                out.append(texts[i])
                i += 1
        return out, proc

    def parse_text(self, lst):
        recs = []
        cur = {}
        ym = None
        now_year = datetime.now().year
        for t in lst:
            t = t.strip()
            ym_m = re.search(r'(\d{4})年(\d{1,2})月', t)
            if ym_m:
                ym = f"{ym_m.group(1)}-{int(ym_m.group(2)):02d}"
                continue
            tm = re.search(r'(?:(\d{1,2})月(\d{1,2})日|(\d{2,4}-\d{2}-\d{2})|(\d{1,2}-\d{2}))\s*(\d{1,2}[:\.]\d{2})', t)
            if tm:
                month, day, full_date, short_date, time = tm.groups()
                if full_date:
                    date = full_date
                elif short_date:
                    m, d = short_date.split('-')
                    date = f"{now_year}-{int(m):02d}-{int(d):02d}"
                else:
                    date = f"{ym.split('-')[0] if ym else now_year}-{int(month):02d}-{int(day):02d}"
                time = time.replace('.', ':')  # 确保时间格式为 HH:MM
                cur['time'] = f"{date} {time}"
                continue
            am = re.search(r'[￥¥]?\s*([+-]?\d+\.?\d*)', t)
            if am:
                val = float(am.group(1))
                if any(w in t for w in ['退款', '返还', '返回']):
                    val = abs(val)
                cur['amount'] = val
            if 'time' in cur:
                item = re.sub(
                    r'\d+[月/-]\d+[日]?.*?\d+:\d+|\d{2}-\d{2}\s*\d{2}:\d{2}|[￥¥+-]\d+\.?\d*|已?[全额部分]退款|返还|[\d\.]+',
                    '', t).strip()
                if item:
                    cur['item'] = cur.get('item', '') + (' ' if cur.get('item') else '') + item
            if all(k in cur for k in ['time', 'amount', 'item']):
                recs.append(cur)
                cur = {}
        return recs

    def chinese_to_arabic(self, text):
        """中文数字转阿拉伯数字"""
        chinese_numbers = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
                           '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
        unit_map = {'十': 10, '百': 100, '千': 1000, '万': 10000}

        if re.match(r'^[+-]?\d+\.?\d*$', text):
            return float(text)

        total = 0
        current = 0
        for char in text:
            if char in chinese_numbers:
                current = chinese_numbers[char]
            elif char in unit_map:
                total += current * unit_map[char]
                current = 0
        return total + current

    def clean_item(self, text):
        """更新清洗逻辑"""
        patterns = [
            r'\d+月\d+日.*?\d+:\d+',  # 日期时间
            r'[￥¥+-]\s?\d+\.?\d*',  # 金额
            r'已?[全额部分]退款|返还',  # 退款标记
            r'[\d\.]+',  # 残留数字
            r'^[\-+*●·\s]+'  # 特殊符号前缀
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return text.strip()[:20]  # 限制项目名称长度

    def select_directory(self):
        """选择图片目录"""
        self.image_dir = filedialog.askdirectory()
        if self.image_dir:
            self.show_thumbnail()

    def show_thumbnail(self):
        # 清掉旧的
        for child in self.thumb_frame.winfo_children():
            child.destroy()

        # 按文件名排序，确保顺序可控
        for f in sorted(os.listdir(self.image_dir)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(self.image_dir, f)
                img = Image.open(path)
                img.thumbnail((100, 100))
                tk_img = ImageTk.PhotoImage(img)

                lbl = ttk.Label(self.thumb_frame, image=tk_img)
                lbl.image = tk_img  # 保留引用，防止被 GC
                lbl.pack(side=tk.LEFT, padx=5, pady=5)

    def process_images(self):
        self.df = pd.DataFrame()
        for f in sorted(os.listdir(self.image_dir)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                lines, _ = self.ocr_image(os.path.join(self.image_dir, f))
                print(f"OCR 行内容：{lines}")
                recs = self.parse_text(lines)
                print(f"解析结果：{recs}")
                self.df = pd.concat([self.df, pd.DataFrame(recs)], ignore_index=True)
        self.update_results()

    def update_results(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        if self.df.empty:
            return
        # 预处理 time 列，确保时间格式正确
        self.df['time'] = self.df['time'].str.replace('.', ':')
        for _, r in self.df.iterrows():
            self.tree.insert('', 'end', values=(r['time'], r['amount'], r['item']))
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        daily = self.df.groupby(pd.to_datetime(self.df['time']).dt.date)['amount'].sum()
        daily.plot(kind='bar', ax=ax)
        ax.set_title("每日消费趋势")
        plt.setp(ax.get_xticklabels(), rotation=45)
        self.canvas.draw()

    def export_data(self):
        """导出数据"""
        save_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel文件", "*.xlsx"), ("CSV文件", "*.csv")]
        )
        if save_path:
            if save_path.endswith('.xlsx'):
                self.df.to_excel(save_path, index=False)
            else:
                self.df.to_csv(save_path, index=False)


if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()
