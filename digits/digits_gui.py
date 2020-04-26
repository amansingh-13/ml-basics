import tkinter as tk
import numpy as np
from tensorflow.keras.models import load_model

img = np.zeros((28, 28))
depiction = [[None for _ in range(28)] for _ in range(28)]
model = load_model('digits/weights.h5')
text_out = ""

def clear():
    for i in range(28):
        for l in range(28):
            img[i][l] = 0
            canvas.delete("all")

def predict():
    upload_img = img.reshape(1, 28, 28, 1)
    prediction_arr = model.predict(upload_img)
    prediction = np.argmax(prediction_arr)
    text_out = "{} with probablity {}".format(prediction, prediction_arr[0][prediction])
    output.delete('1.0',tk.END)
    output.insert(tk.INSERT, "{} with probablity {}".format(prediction, prediction_arr[0][prediction]))

def draw(event):
    column = int(event.x / 10)
    row = int(event.y / 10)
    depiction[row][column] = canvas.create_rectangle(
    column*10, row*10, (column+1)*10, (row+2)*10, fill='white')
    img[row][column] = 0.917618446
    img[row+1][column] = 0.717618446
    for j in [-1,0,1]:
        for k in [-1,0,1]:
            if img[row+j][column+k]<0.7:
                img[row+j][column+k] = 0.443565
                depiction[row+j][column+k] = canvas.create_rectangle(
                (column+k)*10, (row+j)*10, ((column+k)+1)*10, ((row+j)+1)*10, fill='grey')


root = tk.Tk()
root.resizable(0, 0)
canvas = tk.Canvas(root, width=280, height=280, background='black')
canvas.pack()
clearbtn = tk.Button(root, text="Clear", command=clear)
predictbtn = tk.Button(root, text="Predict", command=predict)
output = tk.Text(root,exportselection=0,bg='WHITE',font='Helvetica',height=1,width=20)
clearbtn.pack()
predictbtn.pack()
output.pack()
canvas.bind("<B1-Motion>", draw)

root.mainloop()