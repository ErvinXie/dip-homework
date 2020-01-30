import tkinter as tk
import numpy as np
from tkinter import *
from tkinter.filedialog import askopenfilename, asksaveasfile
import tkinter.messagebox
from PIL import ImageTk
import PIL.Image
from segmentation import *

w = 1440
h = 960
midwidth = 180
picwidth = (w - midwidth) / 2 - 10
picheight = h - 50

root = tk.Tk()
root.title('图像分割')
root.geometry('{}x{}'.format(w, h))
root.resizable(False, False)

origin = None
origin_name = None
result = None

# 显示图片时缩放到适合大小
def resize_to(im, w, h):
    iw = im.size[0]
    ih = im.size[1]

    dw = iw / w
    dh = ih / h
    if dw > dh:
        return im.resize((int(w), int(w / iw * ih)))
    else:
        return im.resize((int(h / ih * iw), int(h)))

# 设置源图片
def set_ori(label):
    selected = askopenfilename()
    if selected == '':
        return
    img_open = PIL.Image.open(selected)
    global origin, origin_name
    origin = np.array(img_open)
    origin_name = selected.split('/')[-1]

    img = ImageTk.PhotoImage(resize_to(img_open, picwidth, picheight))

    label.config(image=img)
    label.image = img  # keep a reference

# 设置输出图片
def set_out(label):
    img_re = PIL.Image.fromarray(result)
    img = ImageTk.PhotoImage(resize_to(img_re, picwidth, picheight))
    label.config(image=img)
    label.image = img

# 保存图片
def save_pic():
    if result is None:
        tk.messagebox.showerror(title='注意', message='没有待保存的图片')
        return
    fname = asksaveasfile(initialfile=origin_name)
    img = PIL.Image.fromarray(result)
    img.save(fname)
    tk.messagebox.showinfo(title='注意', message='图片保存成功')

# GUI
fs = [Frame(root, height=h, width=[picwidth, midwidth, picwidth][i], border=2) for i in range(3)]
for i, f in enumerate(fs):
    f.grid_propagate(False)
    f.grid(row=0, column=i)
    if i == 1:
        continue
    f.grid_rowconfigure(0, weight=1)
    f.grid_rowconfigure(3, weight=1)
    f.grid_columnconfigure(0, weight=1)
    f.grid_columnconfigure(2, weight=1)

pic_ori = Label(fs[0])
pic_ori.grid(row=1, column=1)
Button(fs[0], text='选择图片', command=lambda: set_ori(pic_ori)).grid(row=2, column=1)

pic_out = Label(fs[2])
pic_out.grid(row=1, column=1)
Button(fs[2], text='保存图片', command=lambda: save_pic()).grid(row=2, column=1)

Button(fs[1], text='K聚类', command=lambda: process('kmeans')).pack()
Label(fs[1], text='在下方填写k值').pack()
kmeans_k = Entry(fs[1])
kmeans_k.pack()
Button(fs[1], text='区域增长', command=lambda: process('rGrowth')).pack()
Label(fs[1], text='在下方填写区域阈值').pack()
rg_tresh = Entry(fs[1])
rg_tresh.pack()
Button(fs[1], text='分水岭', command=lambda: process('dam')).pack()
Label(fs[1], text='在下方填写灰度容差').pack()
eps = Entry(fs[1])
eps.pack()

# 决定采用哪一种处理方法
def process(method):
    if origin is None:
        tk.messagebox.showerror(title='注意', message='没有加载图片')
    else:
        if method == 'kmeans':
            k = kmeans_k.get()
            if k == '':
                k = 5
            else:
                try:
                    k = int(k)
                except ValueError:
                    tk.messagebox.showerror(title='注意', message='请填写正确的k值')
                    return
                if k <= 0:
                    tk.messagebox.showerror(title='注意', message='请填写正确的k值')
                    return
                elif k > 16:
                    tk.messagebox.showerror(title='注意', message='k不能超过16')
                    return
            res = kmeans(im=origin, k=k)

        elif method == 'rGrowth':
            t = rg_tresh.get()
            if t == '':
                t = 0.03
            else:
                try:
                    t = float(t)
                except ValueError:
                    tk.messagebox.showerror(title='注意', message='请填写正确的k值')
                    return
            res = regionGrow(im=origin, threshold=t)
        elif method=='dam':
            e = eps.get()
            if e == '':
                e = 0.03
            else:
                try:
                    e = float(e)
                except ValueError:
                    tk.messagebox.showerror(title='注意', message='请填写正确的k值')
                    return

            res = dam(im=origin,EPS=e)
        global result
        result = res
        set_out(pic_out)


root.mainloop()
