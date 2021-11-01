import os
import cv2
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter.ttk import Button,Entry,Label
from PIL import Image,ImageTk
warnings.filterwarnings('ignore')



class GUI:
    def __init__(self):
        self.win = tk.Tk()
        self.win.title("BrainMRI")
        
        app_width = 1000
        app_height = 600
        
        screen_width = self.win.winfo_screenwidth()
        screen_height = self.win.winfo_screenheight()
        
        x = (screen_width/2) - (app_width/2)
        y = (screen_height/2)-(app_height/2)
        
        self.win.geometry(f'{app_width}x{app_height}+{int(x)}+{int(y)}')
        
        self.create_widgets()
        
    def create_widgets(self):

        self.title_font = ('times',12,'bold')
        Label(self.win,text="Brain Cancer Classifier",font=self.title_font).grid(row=1,column=2,pady=3)

        self.loadImage_btn = Button(self.win,text="LoadMRIImage",width=20,command=self.loadMRIImage)
        self.loadImage_btn.grid(row=2,column=2,pady=2)
        self.classify_btn = Button(self.win,text="Classify",state=tk.DISABLED,width=20,command=self.classifyImage)
        self.classify_btn.grid(row=2,column=3,pady=2)
        
        Label(self.win,text="LG Training score").grid(row=4,column=1,sticky = tk.W,padx=3 ,pady = 3)
        self.lg_training_var = tk.StringVar()
        self.lg_training_score = Entry(self.win,textvariable=self.lg_training_var,width=30)
        self.lg_training_score.grid(row=4,column=2,padx=3 ,pady = 3)
        
        Label(self.win,text="LG Testing score").grid(row=4,column=3,sticky = tk.W,padx=3 ,pady = 3)
        self.lg_testing_var = tk.StringVar()
        self.lg_testing_score = Entry(self.win,textvariable=self.lg_testing_var,width=30)
        self.lg_testing_score.grid(row=4,column=4,padx=3 ,pady = 3)

        Label(self.win,text="SVM Training score").grid(row=5,column=1,sticky =tk.W,padx=3 ,pady = 3)
        self.svm_training_var = tk.StringVar()
        self.svm_training_score = Entry(self.win,textvariable=self.svm_training_var,width=30)
        self.svm_training_score.grid(row=5,column=2,padx=3 ,pady = 3)
        
        Label(self.win,text="SVM Testing score").grid(row=5,column=3,sticky = tk.W,padx=3 ,pady = 3)
        self.svm_testing_var = tk.StringVar()
        self.svm_testing_score = Entry(self.win,textvariable=self.svm_testing_var,width=30)
        self.svm_testing_score.grid(row=5,column=4,padx=3 ,pady = 3)

        Label(self.win,text="Tumor Type",font=self.title_font).grid(row=7,column=2,padx=3 ,pady = 3)
        self.resultVar = tk.StringVar()
        self.result = Entry(self.win,textvariable=self.resultVar,width=20)
        self.result.grid(row=7,column=3,padx=3 ,pady = 3)
        
        self.loading = Label(self.win,text='',font=('Helvetica',14,'italic'),foreground="red")
        self.loading.grid(row=8,column=2,pady=6)
        

    def loadMRIImage(self):
        self.classify_btn["state"] = "disabled"
        self.setFieldValues()
        self.loading["text"] = "Loading..."
        self.train_model()
        global img
        file_types = [('Jpg Files','*.jpg')]
        self.filename = askopenfilename(filetypes=file_types)
        img = Image.open(self.filename)
        img_resized = img.resize((400,400))
        img = ImageTk.PhotoImage(img_resized)
        display = Button(self.win,image=img)
        display.grid(row=3,column=2)
        self.loading["text"] = ''
        self.classify_btn["state"] = "normal"
    
    def classifyImage(self):
        #self.loadImage_btn["state"] = "disabled"
        self.loadImage_btn.configure(state="disabled")
        #self.loading["text"] = "Loading..."
        self.loading.configure(text="Loading...")
        #print(self.loading["text"])
        lg = LogisticRegression(C=0.1)
        lg.fit(self.xtrain, self.ytrain)
        
        sv = SVC()
        sv.fit(self.xtrain, self.ytrain)
        pred = sv.predict(self.xtest)
         
        misclassified=np.where(self.ytest!=pred)
        
        dec = {0:'No Tumor', 1:'Positive Tumor'}
        
        img = cv2.imread(self.filename,0)
        img1 = cv2.resize(img, (200,200))
        img1 = img1.reshape(1,-1)/255
        p = sv.predict(img1)
        
        
        #print("Total Misclassified Samples are : ",len(misclassified[0]))
        
        self.setFieldValues(dec[p[0]],
                            lg.score(self.xtrain,self.ytrain),
                            lg.score(self.xtest,self.ytest),
                            sv.score(self.xtrain, self.ytrain),
                            sv.score(self.xtest, self.ytest)
                           )
        self.loading.configure(text = '')
        self.loadImage_btn.configure(state ="normal")
       
        
    def setFieldValues(self,result='',lg_train='',lg_test='',svm_train='',svm_test=''):
        self.resultVar.set(result)
        self.lg_training_var.set(lg_train)
        self.lg_testing_var.set(lg_test)
        self.svm_training_var.set(svm_train)
        self.svm_testing_var.set(svm_test)
        
    def train_model(self):
        path = os.listdir('Training')
        classes = {'no_tumor':0, 'brain_tumor':1 }
        X = []
        Y = []
        for cls in classes:
            pth = 'Training/'+cls
            print(pth)
            for j in os.listdir(pth):
                img = cv2.imread(pth+'/'+j, 0)
                img = cv2.resize(img, (200,200))
                X.append(img)
                Y.append(classes[cls])
        X = np.array(X)
        Y = np.array(Y)
        X_updated = X.reshape(len(X), -1)
        X_updated = X.reshape(len(X), -1)
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(X_updated, Y, random_state=10,test_size=.20)
        self.xtrain = self.xtrain/255
        self.xtest = self.xtest/255
        
gui = GUI()
gui.win.mainloop()