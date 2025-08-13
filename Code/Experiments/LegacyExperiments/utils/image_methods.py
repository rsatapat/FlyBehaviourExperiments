import numpy as np

from pypylon import pylon
import PyCapture2
import time

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
        camera.setConfiguration(grabMode=PyCapture2.GRAB_MODE.DROP_FRAMES, grabTimeout = -1, registerTimeout = 0, registerTimeoutRetries = 0,
                                numBuffers=1, highPerformaceRetrieveBuffer=True)
        print(camera.getConfiguration())
        # print('this')
        #camera.setFormat7Configuration(100, mode=0, offsetX=108, offsetY=0, width=1024, height=1024, pixelFormat=PyCapture2.PIXEL_FORMAT.MONO8)
    elif name.lower() == "basler":
        # conecting to the first available camera
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    else:
        print("No such: "+name+" camera found")
    time.sleep(2)
    return camera

def grab_image(camera,name):
    if name.lower() == 'ptgrey':
        image = camera.retrieveBuffer()
        if image == None:
            image, timestamp = grab_image(camera, name)
        else:
            timestamp = image.getTimeStamp()
            image = np.array(image.getData(), dtype="uint8").reshape((image.getRows(), image.getCols()))
            timestamp = int(timestamp.microSeconds)
    elif name.lower() == 'basler':
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            image = grabResult.GetArray()
            timestamp = grabResult.TimeStamp
        # timestamp = timestamp//1000
        # tstamp_str = str(timestamp)
        # timestamp = int(tstamp_str[4:])
            grabResult.Release()
        else:
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                image = grabResult.GetArray()
                timestamp = grabResult.TimeStamp
                grabResult.Release()
    return image, timestamp