import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter.ttk import Button,Entry,Label
from PIL import Image,ImageTk

window = tk.Tk()

class GUI:
    def __init__(self):
        window.geometry("400x400")
        window.title("BrainMRI")

        self.title_font = ('times',12,'bold')
        Label(window,text="Brain MRI Image",font=self.title_font).grid(row=1,column=2)

        Button(window,text="LoadMRIImage",width=20,command=lambda:self.loadMRIImage()).grid(row=2,column=2)

        Label(window,text="Type of Tumor").grid(row=4,column=3)
        self.resultVar = tk.StringVar()
        self.result = Entry(window,textvariable=self.resultVar,width=20)
        self.result.grid(row=4,column=5)

        window.mainloop()

    def loadMRIImage(self):
        global img
        file_types = [('Jpg Files','*.jpg')]
        filename = askopenfilename(filetypes=file_types)
        img = Image.open(filename)
        #width,height = img.size
        #width_new = int(width/3)
        #height_new = int(height/3)
        img_resized = img.resize((400,400))
        img = ImageTk.PhotoImage(img_resized)
        display = Button(window,image=img)
        display.grid(row=3,column=2)


gui = GUI()
