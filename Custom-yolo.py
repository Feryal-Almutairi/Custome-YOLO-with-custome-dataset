import cv2 as cv
import numpy as np
import  pyttsx3

confThresholold = 0.25
nsmThresholold = 0.40
inpWidth = 416
inpHeight = 416
classesFile = "C:/Users/LENOVO/Downloads/Doors/DoorDetect-Dataset-master/obj.names" 
classes = None

with open(classesFile,'rt') as f :
    classes =f.read().rstrip('\n').split('\n')


modelConf='C:/Users/LENOVO/Downloads/Doors/DoorDetect-Dataset-master/yolo-obj.cfg'
modelWeights='C:/Users/LENOVO/Downloads/Doors/DoorDetect-Dataset-master/yolo-obj.weights'

def postprocess(frame, outs):
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]


    classIDs = []
    confidences = []
    boxes = []



    for out in outs:
        for detection in out:
            scores = detection[5:]
            classID =np.argmax(scores)
            confidence =scores[classID]

            if confidence > confThresholold:
                centerX = int(detection[0]*frameWidth)
                centerY = int(detection[1]*frameWidth)

                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)

                left = int(centerX-width/2)
                top = int(centerY-height/2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes (boxes,confidences, confThresholold, nsmThresholold)
    for i in indices:
     i = i[0]
     box = boxes[i]
     left = box[0]
     top = box[1]
     width = box[2]
     height = box[3]

     drawPred(classIDs[i], confidences[i], left, top, left + width, top + height)
def drawPred(classId,cconf, left,top,right,bottom):
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label ='%.2f' %cconf

    if classes:
        assert(classId<len(classes))
        label='%s:%s' % (classes[classId], label)

    cv.putText(frame, label,(left,top),cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    engine = pyttsx3.init()
    engine.say(classes[classId])
    engine.runAndWait()


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]



net = cv.dnn.readNetFromDarknet(modelConf,modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

winName='Vision system'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
cv.resizeWindow(winName, 1000, 1000)

cap = cv.VideoCapture(0)
while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()

    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop = False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))

    postprocess(frame, outs)

    cv.imshow(winName, frame)
