import numpy as np

from pypylon import pylon
import PyCapture2

def initialise_camera(name):
    '''
    name of camera (str) -> camera object
    :param name:
    :return:
    '''
    if name.lower() == "ptgrey":
        bus = PyCapture2.BusManager()
        numCams = bus.getNumOfCameras()
        camera = PyCapture2.Camera()
        camera.connect(bus.getCameraFromIndex(0))
        camera.startCapture()
    elif name.lower() == "basler":
        # conecting to the first available camera
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    else:
        print("No such: "+name+" camera found")
    return camera

def grab_image(camera,name):
    if name.lower() == 'ptgrey':
        image = camera.retrieveBuffer()
        image = np.array(image.getData(), dtype="uint8").reshape((image.getRows(), image.getCols()))
        timestamp = image.getTimeStamp()
        timestamp = int(timestamp.microSeconds)
    elif name.lower() == 'basler':
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException) #
        if grabResult.GrabSucceeded():
            # Access the image data
            image = grabResult.GetArray()
        timestamp = image.TimeStamp
        timestamp = timestamp//1000
        tstamp_str = str(timestamp)
        timestamp = int(tstamp_str[4:])
    return image, timestamp



    return image

if __name__== "__main__":
    name1 = "ptgrey"
    intialise_camera(name1)