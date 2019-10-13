from tkinter import *
import tkinter as tk
from tkinter.filedialog import askopenfilename
import MIS
import numpy as np
import cv2
import Detection_test

class GUIManager:
    def __init__(self):
        self.master = Tk()
        self.backColor = "DarkSeaGreen1"

        self.master.geometry("415x290")
        self.master.configure(background=self.backColor)

        self.master.title('Medical Images Segmentation')
        self.master.grid()
        self.file_path = ""
        self.maskTreshold = 150
        self.binaryTresh = 45
        self.ans = ""

    def createImage(self, path="im1.tif"):
        img = cv2.imread(path, 0)
        return img

    def openImage(self):
        self.file_path = askopenfilename(
            initialdir=r"C:\Users\n&e\PycharmProjects\ImageProcessing\Medical Images Segmentation",
            filetypes=(("jpg File", ".jpg"), ("All Files", ".*")),
            title="Choose a file.")
        self.file_path = self.file_path.split('/')[-1]

    def showImage(self):
        #global file_path
        img = self.createImage(self.file_path)
        mask = MIS.toBinary(img, 150)
        erodeImage = cv2.erode(mask, np.ones((17, 17), np.uint8), iterations=1)
        binaryImage = MIS.toBinary(img, 45)
        result = MIS.deleteNeuron(binaryImage, erodeImage)

        result = cv2.resize(result, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

        # change colors
        result = cv2.bitwise_not(result)

        contours = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        length1, width1 = MIS.method1(result, contours)
        length2, width2 = Detection_test.method2(result, contours)

        ans = f"method1 - length:+ {length1}, width: {width1} \nmethod2 - length:{length2}+, width: {width2}"
        Label(self.master, bg=self.backColor, fg="medium blue", text=ans).grid(row=11, column=1)

        Label(self.master, bg=self.backColor, text=' ').grid(row=12)  # space


    def init_gui(self):
        Label(self.master, bg=self.backColor, text='              ').grid(row=0, column=0)  # space
        Label(self.master, bg=self.backColor, fg="medium blue",
              text='Hello, please choose the image you want to work on').grid(
            row=0, column=1)

        Label(self.master, bg=self.backColor, text=' ').grid(row=1)  # space

        button1 = tk.Button(self.master, bg="medium blue", bd=6, fg="white", text='Open', width=13, command=self.openImage).grid(
            row=2, column=1)
        Label(self.master, bg=self.backColor, text=' ').grid(row=3)  # space

        Label(self.master, bg=self.backColor, fg="medium blue", text='Choose mask treshold:').grid(row=5, column=1)
        Label(self.master, bg=self.backColor, text=' ').grid(row=6)  # space
        e1 = Entry(self.master , width=8)
        e1.grid(row=5, column=2)
        maskTreshold1 = Entry.get(e1)
        print("maskTreshold1 = ", self.maskTreshold)

        Label(self.master, bg=self.backColor, fg="medium blue", text='Choose binary treshold:').grid(row=7, column=1)
        Label(self.master, bg=self.backColor, text=' ').grid(row=8)  # space

        e2 = Entry(self.master, width=8)
        e2.grid(row=7, column=2)
        binaryTresh1 = Entry.get(e2)
        print("binaryTresh1 = ", self.binaryTresh)

        Label(self.master, bg=self.backColor, text=' ').grid(row=9)  # space

        button2 = tk.Button(self.master, bg="medium blue", bd=6, fg="white", text='Show', width=13, command=self.showImage).grid(
            row=9, column=1)

        Label(self.master, bg=self.backColor, text=' ').grid(row=10)  # space
        mainloop()

if __name__ == "__main__":
    GUIManager().init_gui()
