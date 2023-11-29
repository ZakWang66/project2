import tkinter as tk
from tkinter import Button, ttk
from pandas import DataFrame
import datetime
import time
import json
import threading
from tkinter import messagebox
from tkinter import filedialog, Toplevel, Text


class StartupGUI:
    def __init__(self, root):

        self.steps_description = [
            "加载AI组件...",
            "加载数据库...",
            "导入模型...",
            "启动完成"
        ]

        self.root = root
        self.root.title("")
        self.root.overrideredirect(True)
        self.root.configure(bg='white')


        self.root_width = 800
        self.root_height = 410
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - self.root_width) // 2
        y = (screen_height - self.root_height) // 2
        self.root.geometry(f"{self.root_width}x{self.root_height}+{x}+{y}")


        self.image = tk.PhotoImage(file="load1.png")  # Replace with your image path
        self.image_label = tk.Label(self.root, image=self.image)
        self.image_label.place(x=0, y=0)




        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.root, variable=self.progress_var, maximum=100, length=800
        )
        self.progress_bar.pack(pady=0, side=tk.BOTTOM)


        self.status_label = tk.Label(self.root, text="正在启动...", font=("SimSun", 12))
        self.status_label.configure(bg='black', fg='white')
        self.status_label.place(x=30 ,y=345)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.main_window = None  
        self.startup_thread = threading.Thread(target=self.startup_process)
        self.startup_thread.start()

    def startup_process(self):
        self.main_window = MainWindow(self.update_progress)
        self.root.after(800, self.close_and_show_main_window)

    def close_and_show_main_window(self):
        self.root.withdraw()  
        self.root.after(0, self.set_main_window_focus)

    def set_main_window_focus(self):
        if self.main_window:
            self.main_window.set_focus()

    def update_progress(self, value, step_index):
        self.progress_var.set(value)
        if 0 <= step_index < len(self.steps_description):
            self.status_label.config(text=self.steps_description[step_index])
        self.root.update()

    def update_status(self, text):
        self.status_label.config(text=text)

    def on_close(self):
        self.root.destroy()
        self.root.quit()

class EditableTreeview(ttk.Treeview):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._create_binding()
        self.active_editor = None

    def _create_binding(self):
        self.bind("<Double-Button-1>", self._on_double_click)
        self.winfo_toplevel().bind("<Button-1>", self._on_root_click, add='+')

    def _on_double_click(self, event):
        if self.active_editor:
            self._save_edit()
        row_id = self.identify_row(event.y)
        column_id = self.identify_column(event.x)
        if row_id and column_id and column_id not in ['#1', '#6']:
            self._open_editor(row_id, column_id, event.x, event.y)

    def _open_editor(self, row_id, column_id, x, y):
        bbox = self.bbox(row_id, column_id)
        if bbox:
            entry = tk.Entry(self)
            entry.place(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])
            entry.insert(0, self.set(row_id, column_id))
            entry.focus()
            self.active_editor = (entry, row_id, column_id, bbox)
            entry.bind("<Return>", self._on_return)

    def _on_return(self, event):
        self._save_edit()

    def _on_root_click(self, event):
        if self.active_editor:
            entry, row_id, column, bbox = self.active_editor
            if not (bbox[0] < event.x < bbox[0] + bbox[2] and bbox[1] < event.y < bbox[1] + bbox[3]):
                self._save_edit()

    def _save_edit(self):
        if self.active_editor:
            entry, row_id, column, _ = self.active_editor
            self.set(row_id, column, entry.get())
            entry.destroy()
            self.active_editor = None

class MenuBar:
    def __init__(self, mw):
        self.mw = mw
        self.menubar = tk.Menu(mw.window)
        self.menu1 = tk.Menu(self.menubar, tearoff=0)
        self.menu1.add_command(label="导入数据", command=self.on_open)
        self.menu1.add_command(label="设置", command=self.on_settings)
        self.menu1.add_separator()
        self.menu1.add_command(label="退出", command=mw.window.quit)
        self.menubar.add_cascade(label="菜单", menu=self.menu1)
        mw.window.config(menu=self.menubar)

    def on_open(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Text files", "*.txt")])
        if not file_paths:
            return

        preview_window = Toplevel(self.mw.window)
        preview_window.title("导入数据文件")
        preview_window.iconbitmap('icon.ico') 

        preview_window.grab_set()
        preview_window.transient(self.mw.window)

        root_width = 660
        root_height = 350
        screen_width = preview_window.winfo_screenwidth()
        screen_height = preview_window.winfo_screenheight()
        x = (screen_width - root_width) // 2
        y = (screen_height - root_height) // 2
        preview_window.geometry(f"{root_width}x{root_height}+{x}+{y}")

        preview_window.resizable(False, False)
        preview_window.minsize(1, 1)
        
        frame1 = tk.LabelFrame(preview_window, text="要导入的数据")
        frame1.place(x=10,y=10)
        
        tree = EditableTreeview(frame1, columns=('filename', 'info1', 'info2', 'info3', 'info4', 'info5'), show='headings')
        tree.heading('filename', text='文件名')
        tree.heading('info1', text='检测日期')
        tree.heading('info2', text='投用日期')
        tree.heading('info3', text='部件名称')
        tree.heading('info4', text='材料')
        tree.heading('info5', text='数据点预览')
        tree.column('filename', width=200)
        tree.column('info1', width=60)  
        tree.column('info2', width=60)  
        tree.column('info3', width=150) 
        tree.column('info4', width=50)  
        tree.column('info5', width=100) 

        vsb = ttk.Scrollbar(frame1, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill="y")
        tree.pack(expand=True, fill='both')

        frame2 = tk.Frame(preview_window)
        frame2.place(x=10,y=260)

        tk.Label(frame2, text="已剔除异常值；双击单元格为每条记录填写相关信息，然后点击“导入数据”。", bd=0, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)
        tk.Label(frame2, text="日期格式为六位数字YYMMDD，如已导入过相同文件名的数据，旧数据将会被覆盖”。", bd=0, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

        button = tk.Button(preview_window, text="导入数据", command=lambda: self.on_save(preview_window, tree))
        button.pack(side=tk.BOTTOM, padx=20, pady=10)

        for file_path in file_paths:
            df = self.mw.pd.read_csv(file_path, encoding='gb2312')
            data = self.mw.pd.to_numeric(df[df.columns[0]], errors='coerce')
            data.fillna(method='ffill', inplace=True)
            data.fillna(method='bfill', inplace=True)
            data = data.to_list()
            data = self.mw.pd.DataFrame(data, columns=['data'])
            Q1 = data['data'].quantile(0.25)
            Q3 = data['data'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = data[(data['data'] >= lower_bound) & (data['data'] <= upper_bound)]
            data = data['data'].to_list()
            tree.insert('', 'end', values=(file_path.split('/')[-1], '', '', '', '', str(data)))

    def on_save(self, tw, tree):
        data = self.mw.pd.DataFrame([tree.item(row)["values"] for row in tree.get_children()])
        data2 = self.mw.pd.DataFrame()
        data2['filename'] = data[0].apply(lambda x: str(x))
        data2['date'] = data[1].apply(lambda x: str(x))
        data2['setup_date'] = data[2].apply(lambda x: str(x))
        data2['name'] = data[3].apply(lambda x: str(x))
        data2['material'] = data[4].apply(lambda x: str(x))
        data2['label'] = ''
        data2['model'] = ''
        data2['prediction'] = ''
        data2['raw'] = data[5].apply(lambda x: str(x))
        self.mw.df = self.mw.pd.concat([self.mw.df, data2], ignore_index=True).reset_index(drop=True).drop_duplicates(subset=['filename'], keep='last').reset_index(drop=True)
        self.mw.df.to_csv(self.mw.settings['db'], index=False)
        tw.destroy()

    def on_load_model(self):
        print("Load model selected")

    def on_settings(self):
        preview_window = Toplevel(self.mw.window)
        preview_window.title("设置")
        preview_window.iconbitmap('icon.ico') 
        preview_window.grab_set()
        preview_window.transient(self.mw.window)

        root_width = 660
        root_height = 350
        screen_width = preview_window.winfo_screenwidth()
        screen_height = preview_window.winfo_screenheight()
        x = (screen_width - root_width) // 2
        y = (screen_height - root_height) // 2
        preview_window.geometry(f"{root_width}x{root_height}+{x}+{y}")

        preview_window.resizable(False, False)
        preview_window.minsize(1, 1)
        
        frame0 = tk.LabelFrame(preview_window, text='路径设置', width=200, height=150)
        frame0.pack(padx=10, pady=10, fill="both")

        frame1 = tk.Frame(frame0, width=200, height=100)
        frame1.place(x=10,y=10)
        label10 = tk.Label(frame1, text='数据库路径：', bd=0, relief=tk.SUNKEN, anchor=tk.W)
        self.label_var_db = tk.StringVar()
        self.label_var_db.set(self.mw.settings['db'])
        self.label1 = tk.Label(frame1, textvariable=self.label_var_db, bd=0, relief=tk.SUNKEN, anchor=tk.W)
        button1 = tk.Button(frame1, text="更改", command=lambda: self.on_click1(self.label_var_db))
        label10.pack(padx=5, pady=10, side=tk.LEFT)
        button1.pack(padx=5, pady=10, side=tk.LEFT)
        self.label1.pack(padx=5, pady=10, side=tk.LEFT)

        frame2 = tk.Frame(frame0, width=200, height=100)
        frame2.place(x=10,y=60)
        label20 = tk.Label(frame2, text='模型路径：   ', bd=0, relief=tk.SUNKEN, anchor=tk.W)
        self.label_var_model = tk.StringVar()
        self.label_var_model.set(self.mw.settings['model'])
        label2 = tk.Label(frame2, textvariable=self.label_var_model, bd=0, relief=tk.SUNKEN, anchor=tk.W)
        button2 = tk.Button(frame2, text="更改", command=lambda: self.on_click2(self.label_var_model))
        label20.pack(padx=5, pady=10, side=tk.LEFT)
        button2.pack(padx=5, pady=10, side=tk.LEFT)
        label2.pack(padx=5, pady=10, side=tk.LEFT)

        frame3 = tk.LabelFrame(preview_window, text='腐蚀速率预警线设置（单位mm/y）', width=100, height=150)
        frame3.pack(padx=10, pady=10, side=tk.LEFT)
        validate_input_func = frame3.register(self.validate_input)
        label1 = tk.Label(frame3, text="轻度腐蚀:")
        label1.grid(padx=5, pady=5,row=1, column=0)
        self.entry1 = tk.Entry(frame3, validate="key", validatecommand=(validate_input_func, "%P"))
        self.entry1.grid(padx=5, pady=5,row=1, column=1)
        label2 = tk.Label(frame3, text="中度腐蚀:")
        label2.grid(padx=5, pady=5,row=2, column=0)
        self.entry2 = tk.Entry(frame3, validate="key", validatecommand=(validate_input_func, "%P"))
        self.entry2.grid(padx=5, pady=5,row=2, column=1)
        label3 = tk.Label(frame3, text="严重腐蚀:")
        label3.grid(padx=5, pady=5,row=3, column=0)
        self.entry3 = tk.Entry(frame3, validate="key", validatecommand=(validate_input_func, "%P"))
        self.entry3.grid(padx=5, pady=5,row=3, column=1)

        try:
            self.entry1.insert(0, self.mw.settings['warn_lv1'])
            self.entry2.insert(0, self.mw.settings['warn_lv2'])
            self.entry3.insert(0, self.mw.settings['warn_lv3'])
        except:
            self.entry1.insert(0, '0.6')
            self.entry2.insert(0, '2.6')
            self.entry3.insert(0, '3.8')

        frame4 = tk.Frame(preview_window, width=150, height=150)
        frame4.pack(padx=100, pady=10, side=tk.LEFT)
        button1 = tk.Button(frame4, text='保存', command=lambda: self.on_confirm(preview_window))
        button1.pack(padx=20, pady=5, side=tk.LEFT)
        button2 = tk.Button(frame4, text='取消', command=preview_window.destroy)
        button2.pack(padx=20, pady=5, side=tk.LEFT)

    def on_confirm(self, tw):
        excp = False
        lv1 = self.mw.settings['warn_lv1']
        lv2 = self.mw.settings['warn_lv2']
        lv3 = self.mw.settings['warn_lv3']
        try:
            lv1 = float(self.entry1.get())
            lv2 = float(self.entry2.get())
            lv3 = float(self.entry3.get())
        except:
            excp=True

        try:
            if not excp:
                self.mw.df = self.mw.pd.read_csv(self.label_var_db.get(), dtype=str)
                self.mw.model_default = self.mw.joblib.load(self.label_var_model.get())
        except:
            excp=True
            self.mw.df = self.mw.pd.read_csv(self.mw.settings['db'], dtype=str)
            self.mw.model_default = self.mw.joblib.load(self.mw.settings['model'])

        try:
            if not excp:
                self.mw.filter_data()
                self.mw.settings['db'] =  self.label_var_db.get()
                self.mw.settings['model'] = self.label_var_model.get()
                self.mw.settings['warn_lv1'] = str(lv1)
                self.mw.settings['warn_lv2'] = str(lv2)
                self.mw.settings['warn_lv3'] = str(lv3)
            with open('settings.json', 'w') as file:
                json.dump(self.mw.settings, file)
        except:
            excp=True
        
        tw.grab_release()
        tw.destroy()

    def validate_input(self,P):
        if P.replace(".", "", 1).isdigit() or P == "":
            return True
        else:
            return False

    def on_click1(self, label):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        label.set(file_path)
        
    def on_click2(self, label):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.pkl")])
        if not file_path:
            return
        label.set(file_path)

class MainWindow:
    def get_features(self, x):
        x = self.pd.DataFrame(x).T.iloc[0]

        peaks, _ = self.find_peaks(x)
        num_peaks = len(peaks)
        peak_heights = x[peaks]

        spi = peaks[self.np.argsort(peak_heights)[::-1]]
        sph = self.np.sort(peak_heights)[::-1]
        peak0 = spi[0] if len(peaks)>0 else -1
        peak1 = spi[1] if len(peaks)>1 else -1
        peak2 = spi[2] if len(peaks)>2 else -1
        ph0 = sph[0] if len(peaks)>0 else -1
        ph1 = sph[1] if len(peaks)>1 else -1
        ph2 = sph[2] if len(peaks)>2 else -1
        
        average_slope = (x.iloc[-1] - x.iloc[0]) / len(x)
        waveform_length = (x[1:].reset_index(drop=True) - x[:-1].reset_index(drop=True)).sum()
        mid_point = min(x) + (max(x) - min(x))/2 
        num_zero_crossings = (((x[:-1].reset_index(drop=True) - mid_point) * (x[1:].reset_index(drop=True) - mid_point)) < 0).sum()
    
        curve_length = self.np.sum(self.np.abs(self.np.diff(x)))
        max_autocorrelation = self.np.max(self.correlate(x, x, mode='full')[len(x):])

        
        return [
            (x ** 2).sum(),
            x.sum(),
            num_zero_crossings,
            ph0,
            ph1,
            ph2,
            num_peaks,
            x.std(),
            x.min(),
            self.skew(x), 
            self.kurtosis(x),
            x.mean(),
            x.quantile(0.1),
            x.quantile(0.2),
            x.quantile(0.3),
            x.quantile(0.4),
            x.quantile(0.5),
            x.quantile(0.6),
            x.quantile(0.7),
            x.quantile(0.8),
            x.quantile(0.9),
            self.nolds.sampen(x),
            curve_length,
            max_autocorrelation
        ]

    def set_focus(self):
        self.window.focus_force()

    def on_close(self):
        self.window.destroy()
        tk.Tk().quit()

    def filter_data(self):
        try:
            v1 = self.start_date_entry_var.get()
            v2 = self.end_date_entry_var.get()
            if len(v1)!=6 or len(v2)!=6:
                messagebox.showinfo("输入错误", "请输入有效的日期范围")
                return
            start_date_dt = datetime.datetime.strptime(v1, "%y%m%d")
            end_date_dt = datetime.datetime.strptime(v2, "%y%m%d")

            start_date = start_date_dt.strftime("%y%m%d")
            end_date = end_date_dt.strftime("%y%m%d")

            if start_date and end_date:
                self.filtered_df = self.df[(self.df['date'] >= start_date) & (self.df['date'] <= end_date)]
                self.filtered_df = self.filtered_df[['filename', 'date', 'setup_date', 'name', 'material']].sort_values(['date', 'filename'])


                if self.checkbox_var1.get():
                    keyword = self.search_entry.get().strip().lower()
                    self.filtered_df = self.filtered_df[self.filtered_df['name'].str.contains(keyword, case=False)]
                self.tree.delete(*self.tree.get_children())
                for index, row in self.filtered_df.iterrows():
                    self.tree.insert("", tk.END, iid=index, values=row.tolist())
                self.statusbar.config(text='已加载'+str(len(self.filtered_df))+'条检测记录')
            else:
                messagebox.showinfo("输入错误", "请输入有效的日期范围")
        except:
            messagebox.showinfo("输入错误", "请输入有效的日期范围")

    def on_treeview_select(self, event):

        selected_item = self.tree.selection()
        if selected_item:
            item_data = self.tree.item(selected_item)
            self.target_df = self.df[self.df['filename'] == item_data['values'][0]]
            if len(self.target_df) > 0:

                raw = eval(self.target_df.iloc[0].raw)
                self.ax1.cla()
                self.ax1.plot(raw, label='测量值')
                self.ax1.set_ylim(0, 1.2*max(raw))
                self.ax1.autoscale_view()
                self.canvas1.draw()

                interp = self.np.interp(self.np.linspace(0, len(raw), 200), self.np.arange(len(raw)), raw)
                interp_lowess = self.lowess(interp, range(len(interp)), frac=0.07)[:,1]
                interp = interp/max(interp_lowess)
                interp_lowess2 = interp_lowess/max(interp_lowess)
                self.ax2.cla()
                self.ax2.plot(interp, label='处理值')
                self.ax2.plot(interp_lowess2, label='平滑值')
                x = self.np.argmin(interp_lowess2)
                y = self.np.min(interp_lowess2)
                self.ax2.scatter(x, y, color='red', s=30)
                self.ax2.annotate(f'最小相对厚度\n({x}, {y*100:.1f}%)', xy=(x,y), xytext=(x-10,y-y/15))
                self.ax2.set_ylim(0.9*min(interp), 1.1*max(interp))
                self.ax2.yaxis.set_major_formatter(self.PercentFormatter(1.0))
                self.ax2.autoscale_view()
                self.canvas2.draw()

                res = self.model_default.predict([self.get_features(interp_lowess2)])[0]
                res_proba = self.model_default.predict_proba([self.get_features(interp_lowess2)])[0]
                res_index = self.model_default.classes_

                new_text = '壁厚测量情况（平滑后）：'
                new_text += '最小值：' + "{:.2f}".format(min(interp_lowess)) + ',\t'
                new_text += '最大值：' + "{:.2f}".format(max(interp_lowess)) + '\n'
                
                d_rate = [1.0-i for i in interp_lowess2]
                new_text += '局部缺陷减薄比率：'
                new_text += '平均值：' + "{:.2%}".format(sum(d_rate)/len(d_rate)) + ',\t'
                new_text += '最大值：' + "{:.2%}".format(max(d_rate)) + '\n'
                
                dd = self.target_df.iloc[0]
                date1 = datetime.datetime.strptime(str(dd['setup_date']), "%y%m%d")
                date2 = datetime.datetime.strptime(str(dd['date']), "%y%m%d")
                ygap = (date2 - date1).days/365.25
                c_rate = (max(interp_lowess) - min(interp_lowess)) /ygap
                new_text += '投用至检测时段局部腐蚀速率：' + "{:.3f}".format(c_rate)  +  'mm/y \t'

                try:
                    if c_rate >= float(self.settings['warn_lv3']):
                        warn_lv = '严重腐蚀预警'
                    elif c_rate >= float(self.settings['warn_lv2']):
                        warn_lv = '中度腐蚀预警'
                    elif c_rate >= float(self.settings['warn_lv1']):
                        warn_lv = '轻度腐蚀预警'
                    else:
                        warn_lv = ''
                    new_text += warn_lv + '\n'
                except:
                    pass

                new_text += '\n分析结果：\n模型识别局部缺陷类型：' + res + '\n' 
                if res == '点蚀':
                    new_text += '参考因素：环境因素，温度，侵蚀性离子浓度，PH值等'
                elif res == '磨蚀':
                    new_text += '参考因素：流体特性，材料性能及环境因素，管壁结构工艺、温度等'
                elif res == '开裂':
                    new_text += '参考因素：材料显微组织状态和成分变化及环境中某些化学物质影响'
                elif res == '坑蚀':
                    new_text += '参考因素：介质和环境因素，流体状态、环境温度，及可能的腐蚀物'

                self.model_res.config(state=tk.NORMAL)
                self.model_res.delete("1.0", "end")
                self.model_res.insert("1.0", new_text)
                self.model_res.config(state=tk.DISABLED)

    def delete_selected_item(self):
        selected_item = self.tree.focus()
        if selected_item:
            self.df = self.df.drop(int(selected_item)).reset_index(drop=True)
            self.df.to_csv(self.settings['db'], index=False)
            self.filter_data()

    def on_right_click(self, event):
        iid = self.tree.identify_row(event.y)
        if iid:
            print(iid)
            self.tree.focus(iid)
            self.tree.selection_set(iid)
            self.popup_menu.post(event.x_root, event.y_root)
    
    def validate_date(self, P):
        if P.isdigit() and len(P) <= 6:
            return True
        elif P == "":
            return True
        else:
            return False

    def __init__(self, progress_callback):
        self.progress_callback = progress_callback
        TOTAL_STEPS = 4
        time.sleep(0.5)
        step = 0
        time.sleep(0.1) 
        progress_callback((step + 1) * 100 / TOTAL_STEPS, step)

        try:
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from matplotlib.ticker import PercentFormatter
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import joblib

            from sklearn.ensemble import RandomForestClassifier
            from statsmodels.nonparametric.smoothers_lowess import lowess
            from scipy.signal import find_peaks
            from scipy.stats import skew, kurtosis
            from numpy import correlate
            import nolds
            from sklearn.model_selection import GridSearchCV, cross_val_score

            from matplotlib.font_manager import FontProperties
            font = FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=12)
            plt.rcParams['font.family'] = ['sans-serif']
            plt.rcParams['font.sans-serif'] = [font.get_name()]

            with open('settings.json', 'r') as file:
                self.settings = json.load(file)

            self.np = np
            self.pd = pd
            self.plt = plt
            self.PercentFormatter = PercentFormatter
            self.joblib = joblib
            self.RandomForestClassifier = RandomForestClassifier
            self.lowess = lowess
            self.find_peaks = find_peaks
            self.skew = skew
            self.kurtosis = kurtosis
            self.correlate = correlate
            self.nolds = nolds
            self.GridSearchCV = GridSearchCV
            self.cross_val_score = cross_val_score
            self.Figure =Figure
            self.FigureCanvasTkAgg = FigureCanvasTkAgg
        except:
            messagebox.showerror('错误', "加载AI模块时出错")
            tk.Tk().quit()

        step = 1
        time.sleep(0.1) 
        progress_callback((step + 1) * 100 / TOTAL_STEPS, step)

        try:
            self.df = pd.read_csv(self.settings['db'], dtype=str)
            assert(
                set(self.df.columns).issubset(
                    set(['filename','date','setup_date','name','material','label','model','prediction','raw'])
                    )
            )
        except:
            messagebox.showerror('错误', "读取数据库文件"+ str(self.settings['db']) +"出错")
            tk.Tk().quit()
        step = 2
        time.sleep(0.1) 
        progress_callback((step + 1) * 100 / TOTAL_STEPS, step)

        try:
            self.model_default = joblib.load(self.settings['model'])
        except:
            messagebox.showerror('错误', "读取模型文件"+ str(self.settings['model']) +"出错")
            tk.Tk().quit()

        step = 3
        time.sleep(0.1) 
        progress_callback((step + 1) * 100 / TOTAL_STEPS, step)


        time.sleep(0.3)
        self.window = tk.Toplevel()
        self.window.title("腐蚀缺陷数字模型检测系统")
        self.window.iconbitmap('icon.ico') 
        self.window.lower()

        self.root_width = 1100
        self.root_height = 700
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width - self.root_width) // 2
        y = (screen_height - self.root_height) // 2
        self.window.geometry(f"{self.root_width}x{self.root_height}+{x}+{y}")
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.window.withdraw()

        self.window.resizable(False, False)
        self.window.maxsize(self.window.winfo_screenwidth(), self.window.winfo_screenheight())
        self.window.minsize(1, 1)
        
        self.statusbar = tk.Label(self.window, text="就绪", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        menubar = MenuBar(self)

        self.frame1 = tk.LabelFrame(self.window, text="检索数据")
        self.frame1.place(x=10,y=10)
        
        self.tree = ttk.Treeview(self.frame1, columns=list(self.df.columns[:5]), show="headings", height=25)


        self.popup_menu = tk.Menu(self.tree, tearoff=0)
        self.popup_menu.add_command(label="删除", command=self.delete_selected_item)

        self.tree.bind("<Button-3>", self.on_right_click)

        
        self.tree.bind("<<TreeviewSelect>>", lambda event: self.on_treeview_select(event))
        self.tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=5, pady=5)  

        self.tree.heading('filename', text='文件名')
        self.tree.column('filename', width=200)
        self.tree.heading('date', text='检测日期')
        self.tree.column('date', width=60)
        self.tree.heading('setup_date', text='投用日期')
        self.tree.column('setup_date', width=60)
        self.tree.heading('name', text='部件名称')
        self.tree.column('name', width=150)
        self.tree.heading('material', text='材料')
        self.tree.column('material', width=50)

        vsb = ttk.Scrollbar(self.frame1, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill="y")

        self.filter_frame = tk.Frame(self.window)
        self.filter_frame.place(x=10,y=580)
        
        self.start_date_label = tk.Label(self.filter_frame, text="开始日期:")
        self.start_date_label.pack(side=tk.LEFT)

        validate_input_cmd = root.register(self.validate_date)
        
        self.start_date_entry_var = tk.StringVar()
        self.start_date_entry = tk.Entry(self.filter_frame, textvariable=self.start_date_entry_var, validate="key", validatecommand=(validate_input_cmd, "%P"))
        self.start_date_entry_var.set((datetime.datetime.now() - datetime.timedelta(days=720)).strftime("%y%m%d"))
        self.start_date_entry.pack(side=tk.LEFT)
        
        self.end_date_label = tk.Label(self.filter_frame, text="结束日期:")
        self.end_date_label.pack(side=tk.LEFT)
        
        self.end_date_entry_var = tk.StringVar()
        self.end_date_entry = tk.Entry(self.filter_frame, textvariable=self.end_date_entry_var, validate="key", validatecommand=(validate_input_cmd, "%P"))
        self.end_date_entry_var.set(datetime.datetime.now().strftime("%y%m%d"))
        self.end_date_entry.pack(side=tk.LEFT)

        tk.Label(self.filter_frame, text="(日期格式YYMMDD)").pack(padx=10, side=tk.LEFT)
        
        self.search_frame = tk.Frame(self.window)
        self.search_frame.place(x=10,y=610)

        self.checkbox_var1 = tk.BooleanVar()
        checkbox = tk.Checkbutton(self.search_frame, text='筛选部件名称', variable=self.checkbox_var1)
        checkbox.pack(side=tk.LEFT)

        self.search_entry = tk.Entry(self.search_frame)
        self.search_entry.pack(side=tk.LEFT, padx=5, pady=5)
        self.filter_button = tk.Button(self.search_frame, text="搜索", command=self.filter_data)
        self.filter_button.pack(side=tk.LEFT, padx=15, pady=5)


        self.plt_frame1 = tk.Frame(self.window)
        self.plt_frame1.place(x=580,y=10)

        fig = self.Figure(figsize=(5, 2.4), dpi=100)
        fig.suptitle('测量值')
        self.ax1 = fig.add_subplot(111)
        self.ax1.plot([0], [0])

        self.canvas1 = self.FigureCanvasTkAgg(fig, master=self.plt_frame1)
        canvas_widget1 = self.canvas1.get_tk_widget()
        canvas_widget1.pack(fill=tk.BOTH, expand=True)

        self.plt_frame2 = tk.Frame(self.window)
        self.plt_frame2.place(x=580,y=260)

        fig = self.Figure(figsize=(5, 2.4), dpi=100)
        fig.suptitle('局部缺陷减薄比-重新采样平滑')
        self.ax2 = fig.add_subplot(111)
        self.ax2.plot([0], [0])

        self.canvas2 = self.FigureCanvasTkAgg(fig, master=self.plt_frame2)
        canvas_widget2 = self.canvas2.get_tk_widget()
        canvas_widget2.pack(fill=tk.BOTH, expand=True)


        self.model_frame = tk.LabelFrame(self.window, text='模型分析结果')
        self.model_frame.place(x=580,y=510)
        self.model_res = tk.Text(self.model_frame, wrap=tk.WORD, height=8, width=69)
        self.model_res.insert("1.0", "")
        self.model_res.config(state=tk.DISABLED)
        self.model_res.pack(padx=5, pady=5)

        self.filter_data()
        if self.tree.get_children():
            self.tree.selection_set(self.tree.get_children()[0])
        self.window.deiconify()

if __name__ == "__main__":
    root = tk.Tk()
    app = StartupGUI(root)
    root.mainloop()
