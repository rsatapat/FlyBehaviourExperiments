import os.path
from tkinter import *
from tkinter import ttk
import shelve

global data
data = dict()

global rawdata
rawdata = dict()

def courtship_calculate(*args):
    # if not male_line.get() or not female_line.get() or not repetitions.get() or not male_serial.get() or not female_serial.get():
    #     print("Not all fields have been filled")
    # else:
    print(male_line.get(), female_line.get(), male_age.get(), female_age.get(), male_serial.get(), female_serial.get(), repetitions.get())
    end()
    print('done')
    data = male_line.get(), female_line.get(), male_age.get(), female_age.get(), male_serial.get(), female_serial.get(), repetitions.get(), info.get()
    # end()
    return


def calculate_rectangle(*args):
    if not Fly_line.get() or not Fly_sex.get() or not Fly_age.get() or not Serial_no.get():
        print("Not all fields have been filled")
    else:
        print(Fly_line.get(), Fly_sex.get(), Fly_age.get(), Serial_no.get())
        end()
        print('done')
        data['fly_line'] = Fly_line.get()
        data['fly_sex'] = Fly_sex.get()
        data['fly_age'] = Fly_age.get()
        data['serial_no'] = Serial_no.get()
        data['stimulus_duration'] = float(stim_duration.get())
        data['contrast'] = [float(x) for x in contrast.get().replace(" ", '').split(',')]
        data['remark'] = remark.get()

        rawdata['fly_line'] = Fly_line.get()
        rawdata['fly_sex'] = Fly_sex.get()
        rawdata['fly_age'] = Fly_age.get()
        rawdata['serial_no'] = Serial_no.get()
        rawdata['stimulus_duration'] = stim_duration.get()
        rawdata['contrast'] = contrast.get()
        rawdata['remark'] = remark.get()
        return


def calculate(*args):
    if not Fly_line.get() or not Fly_sex.get() or not Fly_age.get() or not Serial_no.get():
        print("Not all fields have been filled")
    else:
        print(Fly_line.get(), Fly_sex.get(), Fly_age.get(), Serial_no.get())
        end()
        print('done')
        data['fly_line'] = Fly_line.get()
        data['fly_sex'] = Fly_sex.get()
        data['fly_age'] = Fly_age.get()
        data['serial_no'] = Serial_no.get()
        data['stimulus_duration'] = float(stim_duration.get())
        data['contrast'] = [float(x) for x in contrast.get().replace(" ", '').split(',')]
        data['temporal_frequency'] = [float(x) for x in temporal_frequency.get().replace(" ", '').split(',')]
        data['spatial_frequency'] = [float(x) for x in spatial_frequency.get().replace(" ", '').split(',')]
        data['remark'] = remark.get()

        rawdata['fly_line'] = Fly_line.get()
        rawdata['fly_sex'] = Fly_sex.get()
        rawdata['fly_age'] = Fly_age.get()
        rawdata['serial_no'] = Serial_no.get()
        rawdata['stimulus_duration'] = stim_duration.get()
        rawdata['contrast'] = contrast.get()
        rawdata['temporal_frequency'] = temporal_frequency.get()
        rawdata['spatial_frequency'] = spatial_frequency.get()
        rawdata['remark'] = remark.get()
        return


def calculate_half_pinwheel(*args):
    if not Fly_line.get() or not Fly_sex.get() or not Fly_age.get() or not Serial_no.get():
        print("Not all fields have been filled")
    else:
        print(Fly_line.get(), Fly_sex.get(), Fly_age.get(), Serial_no.get())
        end()
        print('done')
        data['fly_line'] = Fly_line.get()
        data['fly_sex'] = Fly_sex.get()
        data['fly_age'] = Fly_age.get()
        data['serial_no'] = Serial_no.get()
        data['stimulus_duration'] = float(stim_duration.get())
        data['contrast'] = []
        data['temporal_frequency'] = []
        data['spatial_frequency'] = []
        data['contrast'].append([float(x) for x in contrast.get().replace(" ", '').split(';')[0].split(',')])
        data['contrast'].append([float(x) for x in contrast.get().replace(" ", '').split(';')[1].split(',')])
        data['temporal_frequency'].append([float(x) for x in temporal_frequency.get().replace(" ", '').split(';')[0].split(',')])
        data['temporal_frequency'].append([float(x) for x in temporal_frequency.get().replace(" ", '').split(';')[1].split(',')])
        data['spatial_frequency'].append([float(x) for x in spatial_frequency.get().replace(" ", '').split(';')[0].split(',')])
        data['spatial_frequency'].append([float(x) for x in spatial_frequency.get().replace(" ", '').split(';')[1].split(',')])
        data['remark'] = remark.get()

        rawdata['fly_line'] = Fly_line.get()
        rawdata['fly_sex'] = Fly_sex.get()
        rawdata['fly_age'] = Fly_age.get()
        rawdata['serial_no'] = Serial_no.get()
        rawdata['stimulus_duration'] = stim_duration.get()
        rawdata['contrast'] = contrast.get()
        rawdata['temporal_frequency'] = temporal_frequency.get()
        rawdata['spatial_frequency'] = spatial_frequency.get()
        rawdata['remark'] = remark.get()
        return


def calculate_dots(*args):
    if not Fly_line.get() or not Fly_sex.get() or not Fly_age.get() or not Serial_no.get():
        print("Not all fields have been filled")
    else:
        print(Fly_line.get(), Fly_sex.get(), Fly_age.get(), Serial_no.get())
        end()
        print('done')
        data['fly_line'] = Fly_line.get()
        data['fly_sex'] = Fly_sex.get()
        data['fly_age'] = Fly_age.get()
        data['serial_no'] = Serial_no.get()
        data['stimulus_duration'] = float(stim_duration.get())
        data['contrast'] = [float(x) for x in contrast.get().replace(" ", '').split(',')]
        data['N_dots'] = [float(x) for x in N_dots.get().replace(" ", '').split(',')]
        data['rod'] = [float(x) for x in rod.get().replace(" ", '').split(',')]
        data['coherence'] = [float(x) for x in coherence.get().replace(" ", '').split(',')]
        data['remark'] = remark.get()

        rawdata['fly_line'] = Fly_line.get()
        rawdata['fly_sex'] = Fly_sex.get()
        rawdata['fly_age'] = Fly_age.get()
        rawdata['serial_no'] = Serial_no.get()
        rawdata['stimulus_duration'] = stim_duration.get()
        rawdata['contrast'] = contrast.get()
        rawdata['N_dots'] = N_dots.get()
        rawdata['rod'] = rod.get()
        rawdata['coherence'] = coherence.get()
        rawdata['remark'] = remark.get()
        return


def end():
    root.destroy()
    return


def pinwheel_info(type_of_experiment):
    global root
    root = Tk()
    root.title("Experimental Conditions")

    # shape and size etc.
    mainframe = ttk.Frame(root, padding="3 3 12 12")
    mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
    mainframe.columnconfigure(0, weight=1)
    mainframe.rowconfigure(0, weight=1)

    global Fly_line
    global Fly_sex
    global Fly_age
    global Serial_no
    global remark
    global stim_duration
    global contrast
    global temporal_frequency
    global spatial_frequency


    default_filename = type_of_experiment+'_defaults.dat'
    print(default_filename)
    if os.path.exists(default_filename):
        print('default file exists..')
        db = shelve.open(type_of_experiment+'_defaults')
        Fly_line = StringVar(value=db['fly_line'])
        Fly_sex = StringVar(value=db['fly_sex'])
        Fly_age = StringVar(value=db['fly_age'])
        Serial_no = StringVar(value=db['serial_no'])
        stim_duration = StringVar(value=db['stimulus_duration'])
        contrast = StringVar(value=db['contrast'])
        temporal_frequency = StringVar(value=db['temporal_frequency'])
        spatial_frequency = StringVar(value=db['spatial_frequency'])
        remark = StringVar(value=db['remark'])
    else:
        print('here')
        db = shelve.open(type_of_experiment + '_defaults')
        Fly_line = StringVar()
        Fly_sex = StringVar()
        Fly_age = StringVar()
        Serial_no = StringVar()
        stim_duration = StringVar()
        contrast = StringVar()
        temporal_frequency = StringVar()
        spatial_frequency = StringVar()
        remark = StringVar()

    feet_entry = ttk.Entry(mainframe, width=10, textvariable=Fly_line)
    feet_entry.grid(column=2, row=1, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=Serial_no)
    feet_entry.grid(column=2, row=2, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=Fly_sex)
    feet_entry.grid(column=2, row=3, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=Fly_age)
    feet_entry.grid(column=2, row=4, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=stim_duration)
    feet_entry.grid(column=2, row=5, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=contrast)
    feet_entry.grid(column=2, row=6, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=temporal_frequency)
    feet_entry.grid(column=2, row=7, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=spatial_frequency)
    feet_entry.grid(column=2, row=8, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=remark)
    feet_entry.grid(column=2, row=9, sticky=(E))
    
    if type_of_experiment=='Full pinwheel' or type_of_experiment=='Flicker pinwheel':
        ttk.Button(mainframe, text="Done!", command=calculate).grid(column=3, row=8, sticky=W)
    elif type_of_experiment=='Half pinwheel' or type_of_experiment=='Half pinwheel ring' or type_of_experiment=='Half pinwheel flicker'\
        or type_of_experiment=='Half pinwheel binocular overlap' or type_of_experiment=='Half random dots' or type_of_experiment=='Quarter pinwheel front' or \
    type_of_experiment=='Half pinwheel oscillate':
        ttk.Button(mainframe, text="Done!", command=calculate_half_pinwheel).grid(column=3, row=8, sticky=W)

    ttk.Label(mainframe, text="Line").grid(column=1, row=1, sticky=W)
    ttk.Label(mainframe, text="Serial").grid(column=1, row=2, sticky=W)
    ttk.Label(mainframe, text="Sex").grid(column=1, row=3, sticky=W)
    ttk.Label(mainframe, text="Age").grid(column=1, row=4, sticky=W)
    ttk.Label(mainframe, text="Stimulus duration").grid(column=1, row=5, sticky=W)
    ttk.Label(mainframe, text="Contrast").grid(column=1, row=6, sticky=W)
    ttk.Label(mainframe, text="Temporal Frequency").grid(column=1, row=7, sticky=W)
    ttk.Label(mainframe, text="Spatial Frequency").grid(column=1, row=8, sticky=W)
    ttk.Label(mainframe, text="Remarks(in name)").grid(column=1, row=9, sticky=W)

    for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)

    feet_entry.focus()
    root.mainloop()
    print(data)
    db.update(rawdata)
    db.close()
    return data


def dots_info(type_of_experiment):
    global root
    root = Tk()
    root.title("Experimental Conditions")

    # shape and size etc.
    mainframe = ttk.Frame(root, padding="3 3 12 12")
    mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
    mainframe.columnconfigure(0, weight=1)
    mainframe.rowconfigure(0, weight=1)

    global Fly_line
    global Fly_sex
    global Fly_age
    global Serial_no
    global remark
    global stim_duration
    global contrast
    global N_dots
    global rod
    global coherence


    default_filename = type_of_experiment+'_defaults.dat'
    print(default_filename)
    if os.path.exists(default_filename):
        print('default file exists..')
        db = shelve.open(type_of_experiment+'_defaults')
        Fly_line = StringVar(value=db['fly_line'])
        Fly_sex = StringVar(value=db['fly_sex'])
        Fly_age = StringVar(value=db['fly_age'])
        Serial_no = StringVar(value=db['serial_no'])
        stim_duration = StringVar(value=db['stimulus_duration'])
        contrast = StringVar(value=db['contrast'])
        N_dots = StringVar(value=db['N_dots'])
        rod = StringVar(value=db['rod'])
        coherence = StringVar(value=db['coherence'])
        remark = StringVar(value=db['remark'])
    else:
        db = shelve.open(type_of_experiment + '_defaults')
        Fly_line = StringVar()
        Fly_sex = StringVar()
        Fly_age = StringVar()
        Serial_no = StringVar()
        stim_duration = StringVar()
        contrast = StringVar()
        N_dots = StringVar()
        rod = StringVar()
        coherence = StringVar()
        remark = StringVar()

    feet_entry = ttk.Entry(mainframe, width=10, textvariable=Fly_line)
    feet_entry.grid(column=2, row=1, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=Serial_no)
    feet_entry.grid(column=2, row=2, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=Fly_sex)
    feet_entry.grid(column=2, row=3, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=Fly_age)
    feet_entry.grid(column=2, row=4, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=stim_duration)
    feet_entry.grid(column=2, row=5, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=contrast)
    feet_entry.grid(column=2, row=6, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=N_dots)
    feet_entry.grid(column=2, row=7, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=rod)
    feet_entry.grid(column=2, row=8, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=coherence)
    feet_entry.grid(column=2, row=9, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=remark)
    feet_entry.grid(column=2, row=10, sticky=(E))

    ttk.Button(mainframe, text="Done!", command=calculate_dots).grid(column=3, row=8, sticky=W)

    ttk.Label(mainframe, text="Line").grid(column=1, row=1, sticky=W)
    ttk.Label(mainframe, text="Serial").grid(column=1, row=2, sticky=W)
    ttk.Label(mainframe, text="Sex").grid(column=1, row=3, sticky=W)
    ttk.Label(mainframe, text="Age").grid(column=1, row=4, sticky=W)
    ttk.Label(mainframe, text="Stimulus duration").grid(column=1, row=5, sticky=W)
    ttk.Label(mainframe, text="Contrast").grid(column=1, row=6, sticky=W)
    ttk.Label(mainframe, text="no_of_dots").grid(column=1, row=7, sticky=W)
    ttk.Label(mainframe, text="rate of disappearance").grid(column=1, row=8, sticky=W)
    ttk.Label(mainframe, text="coherence").grid(column=1, row=9, sticky=W)
    ttk.Label(mainframe, text="Remarks(in name)").grid(column=1, row=10, sticky=W)

    for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)

    feet_entry.focus()
    root.mainloop()
    print(data)
    db.update(rawdata)
    db.close()
    return data


def courtship_info():
    type_of_experiment = 'courtship'
    db = shelve.open(type_of_experiment+'_defaults.dat')
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

    for child in mainframe.winfo_children():
        child.grid_configure(padx=5, pady=5)

    feet_entry.focus()
    root.mainloop() # blocks further execution till root has been destroyed by the end() function
    print(data)
    return data


def rectangle_info(type_of_experiment):
    global root
    root = Tk()
    root.title("Experimental Conditions")

    # shape and size etc.
    mainframe = ttk.Frame(root, padding="3 3 12 12")
    mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
    mainframe.columnconfigure(0, weight=1)
    mainframe.rowconfigure(0, weight=1)

    global Fly_line
    global Fly_sex
    global Fly_age
    global Serial_no
    global remark
    global stim_duration
    global contrast


    default_filename = type_of_experiment+'_defaults.dat'
    print(default_filename)
    if os.path.exists(default_filename):
        print('default file exists..')
        db = shelve.open(type_of_experiment+'_defaults')
        Fly_line = StringVar(value=db['fly_line'])
        Fly_sex = StringVar(value=db['fly_sex'])
        Fly_age = StringVar(value=db['fly_age'])
        Serial_no = StringVar(value=db['serial_no'])
        stim_duration = StringVar(value=db['stimulus_duration'])
        contrast = StringVar(value=db['contrast'])
        remark = StringVar(value=db['remark'])
    else:
        print('here')
        db = shelve.open(type_of_experiment + '_defaults')
        Fly_line = StringVar()
        Fly_sex = StringVar()
        Fly_age = StringVar()
        Serial_no = StringVar()
        stim_duration = StringVar()
        contrast = StringVar()
        remark = StringVar()

    feet_entry = ttk.Entry(mainframe, width=10, textvariable=Fly_line)
    feet_entry.grid(column=2, row=1, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=Serial_no)
    feet_entry.grid(column=2, row=2, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=Fly_sex)
    feet_entry.grid(column=2, row=3, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=Fly_age)
    feet_entry.grid(column=2, row=4, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=stim_duration)
    feet_entry.grid(column=2, row=5, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=contrast)
    feet_entry.grid(column=2, row=6, sticky=(E))
    feet_entry = ttk.Entry(mainframe, width=10, textvariable=remark)
    feet_entry.grid(column=2, row=7, sticky=(E))
    
    ttk.Button(mainframe, text="Done!", command=calculate_rectangle).grid(column=3, row=8, sticky=W)

    ttk.Label(mainframe, text="Line").grid(column=1, row=1, sticky=W)
    ttk.Label(mainframe, text="Serial").grid(column=1, row=2, sticky=W)
    ttk.Label(mainframe, text="Sex").grid(column=1, row=3, sticky=W)
    ttk.Label(mainframe, text="Age").grid(column=1, row=4, sticky=W)
    ttk.Label(mainframe, text="Stimulus duration").grid(column=1, row=5, sticky=W)
    ttk.Label(mainframe, text="Contrast").grid(column=1, row=6, sticky=W)
    ttk.Label(mainframe, text="Remarks(in name)").grid(column=1, row=7, sticky=W)

    for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)

    feet_entry.focus()
    root.mainloop()
    print(data)
    db.update(rawdata)
    db.close()
    return data


def get_stimulus_type():
    newroot = Tk()
    newroot.title("Type of Stimulus")

    mainframe = ttk.Frame(newroot, padding="3 3 12 12")
    mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
    mainframe.columnconfigure(0, weight=1)
    mainframe.rowconfigure(0, weight=1)

    stimulus = StringVar(newroot, "Full pinwheel")
    stimulus_types = ["Full pinwheel", "Half pinwheel", "Quarter Pinwheel", "Full Random Dots", "Half Opposing Dots", "Full Opposing Dots",
                      "Half pinwheel flicker", 'Flicker pinwheel', 'Half pinwheel ring', 'Half pinwheel binocular overlap',
                      'Half random dots', "Quarter pinwheel front",'Half pinwheel oscillate', 'dark and light']

    for i, value in enumerate(stimulus_types):
        Radiobutton(mainframe, text=value, variable=stimulus, value=value).grid(column=1, row=i, sticky=W)

    Button(mainframe, text="Done!", command=newroot.destroy).grid(column=1, row=i+1, sticky=W)
    newroot.mainloop()

    return stimulus.get()

def get_experiment_and_stimulus_parameters(stimulus_type):
    if stimulus_type == 'Full pinwheel' or stimulus_type == 'Flicker pinwheel':
        print(stimulus_type)
        data = pinwheel_info(stimulus_type)
    elif stimulus_type == 'Half pinwheel' or stimulus_type == 'Half pinwheel ring' or stimulus_type == 'Half pinwheel flicker'\
        or stimulus_type == 'Half pinwheel binocular overlap' or stimulus_type == "Quarter pinwheel front" or stimulus_type == 'Half pinwheel oscillate':
        print(stimulus_type)
        data = pinwheel_info(stimulus_type)
    elif stimulus_type == "Full Random Dots" or  stimulus_type == "Full Opposing Dots":
        print(stimulus_type)
        data = dots_info(stimulus_type)
    elif stimulus_type == "Half random dots":
        print(stimulus_type)
        data = dots_info(stimulus_type)
    elif stimulus_type == "dark and light":
        print(stimulus_type)
        data = rectangle_info(stimulus_type)
    return data