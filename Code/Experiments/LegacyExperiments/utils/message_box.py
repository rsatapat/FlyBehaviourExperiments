import tkinter as tk
from tkinter import messagebox

def data_change_error(title="Random",message='Random'):
    root = tk.Tk()
    root.title("Data change warning")

    canvas1 = tk.Canvas(root, width=800, height=350)
    canvas1.pack()
    MsgBox = tk.messagebox.askquestion(title, message,
                                       icon='warning')
    if MsgBox == 'yes':
        root.destroy()
        return 1
    else:
        root.destry()
        return 0

if __name__ == '__main__':
    data_change_error()