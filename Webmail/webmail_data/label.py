from tkinter import *
from PIL import ImageTk, Image
import os
import numpy as np

start = 2001
dictionary = {};
curr = start;
master = Tk()
e = Entry(master)
e.pack()
e.focus_set()
path = str(start) + ".jpg"
img = ImageTk.PhotoImage(Image.open(path))
panel = Label(master, image = img)
panel.pack(side = "top", fill = "both", expand = "yes")

def callback():
	global curr, dictionary;
	hello = e.get() # This is the text you may want to use later
	e.delete(0, END)
	print (hello)
	temp = {curr:hello};
	dictionary.update(temp);
	curr += 1;
	print(curr)
	path = str(curr) + ".jpg"
	img2 = ImageTk.PhotoImage(Image.open(path))
	panel.configure(image=img2)
	panel.image = img2

master.bind("<Return>", lambda event: callback())
b = Button(master, text = "OK", width = 10, command = callback)
b.pack()
mainloop()
print (curr-1);
np.save('label'+str(start)+'_'+str(curr-1)+'.npy', dictionary)

#read_dictionary = np.load('my_file.npy').item()
