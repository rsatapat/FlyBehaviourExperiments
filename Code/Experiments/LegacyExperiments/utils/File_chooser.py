import os
from tkinter import Tk, filedialog
import shelve

def file_chooser():
    default_filename = 'most_recent_folder.dat'
    if os.path.exists(default_filename):
        db = shelve.open('most_recent_folder')
        Dir = db['folder']
    else:
        db = shelve.open('most_recent_folder')
        Dir = os.getcwd()
    application_window = Tk()
    # Ask the user to select a folder.
    answer = filedialog.askdirectory(parent=application_window, initialdir=Dir, title="Please select a folder:")
    db.update(dict(folder=answer))
    db.close()
    return answer