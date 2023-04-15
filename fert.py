from tkinter import *
import tkinter as tk
import time
import serial
import threading
import continuous_threading
def f1():
    import pandas as pd
    d = pd.read_csv('To_Use_data.csv')
    import numpy as np
    label = np.array(d['labels'])
    d= d.drop('labels', axis = 1)
    d_list = list(d.columns)
    d = np.array(d)
    from sklearn.model_selection import train_test_split
    train_features, test_features, train_labels, test_labels = train_test_split(d, label, test_size = 0.25, random_state =42)
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_features, train_labels)
    nit=entnit.get()
    pho=entpho.get()
    pot=entpot.get()
    temp= enttemp.get()
    humid= enthumid.get()
    rain= entrain.get()
    ph= entph.get()
    testfeatures = [[nit,pho,pot,temp,humid,rain,ph]]
    predictions = rf.predict(testfeatures)
    entres = Entry(root, bd=8)
    entres.insert(INSERT, predictions[0])
    entres.place(x=240, y=400)

def f2():
    import pandas as pd
    d = pd.read_csv('fert.csv')
    import numpy as np
    label = np.array(d['fertilizers'])
    d= d.drop('fertilizers', axis = 1)
    d_list = list(d.columns)
    d = np.array(d)
    from sklearn.model_selection import train_test_split
    train_features, test_features, train_fertilizers, test_fertilizers = train_test_split(d, label, test_size = 0.25, random_state =42)
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_features, train_fertilizers)
    nit=entnit.get()
    pho=entpho.get()
    pot=entpot.get()
    temp= enttemp.get()
    humid= enthumid.get()
    rain= entrain.get()
    ph= entph.get()
    testfeatures = [[nit,pho,pot,temp,humid,rain,ph]]
    predictions = rf.predict(testfeatures)
    entres1 = Entry(root, bd=8)
    entres1.insert(INSERT, predictions[0])
    entres1.place(x=240, y=400)
ser = serial.Serial('COM3',4800)
vall = 0
index = []
def readserial():
    global vall
    ser_bytes = ser.readline()
    ser_bytes = ser_bytes.decode("utf-8")
    vall = ser_bytes

    index.append(vall)
    

    if len(index) == 1:
        disp1 = tk.Label(root,text=index[0]).place(x=50,y=10)
        
    elif len(index) == 2:
        disp1 = tk.Label(root,text=index[1]).place(x=50,y=40)
        print(index[0])
        print(index[1])

    
    else:
        index.clear()
        
        
    

t1 = continuous_threading.PeriodicThread(0.5,readserial)


root = tk.Tk()
root.geometry("300x250")
tem = tk.Label(root,text='Temp.').place(x=10,y=10)
hum = tk.Label(root,text='Hum.').place(x=10,y=40)
t1.start()
root.mainloop()

print(vall[0]);
