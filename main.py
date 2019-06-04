import tkinter as tk
import tkinter.filedialog as tkFiledialog
from tkinter import font as tkFont
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from blob_detector import detect_blobs
from getting_lines import get_staffs
from note import *
from photo_adjuster import adjust_photo

global file_path

def select_file():
    file_path = tkFiledialog.askopenfilename(initialdir='./', title='Select an Image', filetypes=(("jpeg", "*.jpeg"), ("jpg", "*.jpg"), ("png", "*.png")))
    if file_path:
        filename = os.path.basename(file_path)
        text = tkFont.Font(family="Helvetica", size=18)
        button.configure(text=filename, wraplength=text.measure(filename))
        image = cv.imread(file_path)
        adjusted_photo = adjust_photo(image)
        staffs = get_staffs(adjusted_photo)
        blobs = detect_blobs(adjusted_photo, staffs)
        notes = extract_notes(blobs, staffs, adjusted_photo)
        print(notes)

main_window = tk.Tk()
main_window.geometry('300x150')
main_window.title('Main Page')

# define frame
middle_frame = tk.Frame(main_window)
middle_frame.place(rely=0.1, relwidth=1, relheight=1)

# label
label = tk.Label(middle_frame, text='Please choose a music sheet', font=("Helvetica", 18))
label.grid(row=0, sticky='w', padx=16, pady=16)

# button
button = tk.Button(middle_frame, text='Select File', font=("Helvetica", 18), command=select_file)
button.grid(row=1, sticky='nsew', padx=16, pady=16)

main_window.mainloop()