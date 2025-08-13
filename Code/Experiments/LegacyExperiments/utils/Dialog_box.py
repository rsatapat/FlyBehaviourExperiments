from tkinter import *
from tkinter import ttk
import shelve


def courtship_calculate(*args):
    # if not male_line.get() or not female_line.get() or not repetitions.get() or not male_serial.get() or not female_serial.get():
    #     print("Not all fields have been filled")
    # else:
        print (male_line.get(),female_line.get(),male_age.get(),female_age.get(),male_serial.get(),female_serial.get(),repetitions.get())
        end()
        print('done')
        global data
        data = male_line.get(),female_line.get(),male_age.get(),female_age.get(),male_serial.get(),female_serial.get(),repetitions.get(), info.get()
        #end()
        return

def calculate(*args):
    if not Fly_line.get() or not Fly_sex.get() or not Fly_age.get() or not Serial_no.get():
        print("Not all fields have been filled")
    else:
        print (Fly_line.get(),Fly_sex.get(),Fly_age.get(),Serial_no.get())
        end()
        print('done')
        global data
        data = Fly_line.get(),Fly_sex.get(),Fly_age.get(),Serial_no.get(),bg.get(),fg.get(), info.get(), stim_duration.get()
        #end()
        return

def end():
    root.destroy()
    return

def info():
    db = shelve.open('defaults.dat')

    global root
    root = Tk()
    root.title("Experimental Conditions")

    #shape and size etc.
    mainframe = ttk.Frame(root, padding="3 3 12 12")
    mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
    mainframe.columnconfigure(0, weight=1)
    mainframe.rowconfigure(0, weight=1)

    global Fly_line
    global Fly_sex
    global Fly_age
    global Serial_no
    global bg
    global fg
    global info
    global stim_duration
    
    Fly_line = StringVar()
    Fly_sex = StringVar()
    Fly_age = StringVar()
    Serial_no = StringVar()
    bg = StringVar()
    fg = StringVar()
    info = StringVar()
    stim_duration = StringVar()

    feet_entry = ttk.Entry(mainframe, width=10, textvariable=Fly_line)
    feet_entry.grid(column=2, row=1, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=Serial_no)
    feet_entry.grid(column=2, row=2, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=Fly_sex)
    feet_entry.grid(column=2, row=3, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=Fly_age)
    feet_entry.grid(column=2, row=4, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=stim_duration)
    feet_entry.grid(column=2, row=7, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=info)
    feet_entry.grid(column=2, row=8, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=bg)
    feet_entry.grid(column=2, row=5, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=fg)
    feet_entry.grid(column=2, row=6, sticky=(E))

    ttk.Button(mainframe, text="Done!", command=calculate).grid(column=3, row=8, sticky=W)

    ttk.Label(mainframe, text="Line").grid(column=1, row=1, sticky=W)
    ttk.Label(mainframe, text="Serial").grid(column=1, row=2, sticky=W)
    ttk.Label(mainframe, text="Sex").grid(column=1, row=3, sticky=W)
    ttk.Label(mainframe, text="Age").grid(column=1, row=4, sticky=W)
    ttk.Label(mainframe, text="Stimulus duration").grid(column=1, row=7, sticky=W)
    ttk.Label(mainframe, text="Remarks(in name)").grid(column=1, row=8, sticky=W)
    ttk.Label(mainframe, text="Background").grid(column=1, row=5, sticky=W)
    ttk.Label(mainframe, text="Stimulus").grid(column=1, row=6, sticky=W)

    for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)

    feet_entry.focus()
    root.mainloop()
    print(data)
    return data


def courtship_info():
    global root
    root = Tk()
    root.title("Experimental Conditions")

    # shape and size etc.
    mainframe = ttk.Frame(root, padding="3 3 12 12")
    mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
    mainframe.columnconfigure(0, weight=1)
    mainframe.rowconfigure(0, weight=1)

    global male_line
    global female_line
    global male_age
    global female_age
    global male_serial
    global female_serial
    global repetitions
    global info

    male_line = StringVar()
    female_line = StringVar()
    male_age = StringVar()
    female_age = StringVar()
    male_serial = StringVar()
    female_serial = StringVar()
    repetitions = StringVar()
    info = StringVar()

    feet_entry = ttk.Entry(mainframe, width=10, textvariable=male_line)
    feet_entry.grid(column=2, row=1, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=female_line)
    feet_entry.grid(column=2, row=2, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=male_age)
    feet_entry.grid(column=2, row=3, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=female_age)
    feet_entry.grid(column=2, row=4, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=male_serial)
    feet_entry.grid(column=2, row=5, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=female_serial)
    feet_entry.grid(column=2, row=6, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=repetitions)
    feet_entry.grid(column=2, row=7, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=info)
    feet_entry.grid(column=2, row=8, sticky=(E))

    # ttk.Label(mainframe, textvariable=meters).grid(column=2, row=2, sticky=(W, E))
    ttk.Button(mainframe, text="Done!", command=courtship_calculate).grid(column=3, row=8, sticky=W)

    ttk.Label(mainframe, text="male_line").grid(column=1, row=1, sticky=W)
    ttk.Label(mainframe, text="female_line").grid(column=1, row=2, sticky=W)
    ttk.Label(mainframe, text="male_age").grid(column=1, row=3, sticky=W)
    ttk.Label(mainframe, text="female_age").grid(column=1, row=4, sticky=W)
    ttk.Label(mainframe, text="Male serial").grid(column=1, row=5, sticky=W)
    ttk.Label(mainframe, text="Female serial").grid(column=1, row=6, sticky=W)
    ttk.Label(mainframe, text="Duration (in mins)").grid(column=1, row=7, sticky=W)
    ttk.Label(mainframe, text="Remarks(in name)").grid(column=1, row=8, sticky=W)

    for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)

    feet_entry.focus()
    root.mainloop()
    print(data)
    return data
