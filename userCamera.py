import numpy as np
import pickle
import cv2
import tkinter as Tk
import gc

def loadDictionary():
    pickle_in=open("faces.pickle","rb")
    dictionaryofnames=pickle.load(pickle_in)
    pickle_in.close()
    return dictionaryofnames

def rewriteDictionary():
    pickle_out = open("faces.pickle","wb")
    pickle.dump(names, pickle_out)
    pickle_out.close()


def clearRecords():
    global facenumber
    global names
    global dataf
    global labels
    pickle_out = open("faces.pickle","wb")
    pickle.dump({}, pickle_out)
    pickle_out.close()
    facenumber=0
    names={}
    dataf=[]
    labels=[]


def loadLists(names):
    if(len(names)==0):
        return ([],[])
    filelist=[]
    for i in range(len(names)):
        npfname="FACE"+str(i)+'.npy'
        tmpobj=np.load(npfname).reshape((20,50*50*3)) #FACE i
        filelist.append(tmpobj)
    labels=np.zeros((20*len(names),1))
    for i in range(len(names)):
        for j in range(20):
            labels[j+20*i]=float(i)
    listoffiles=[]
    for i in range(len(names)):
        listoffiles.append(filelist[i])
    if(len(listoffiles)!=0):
        dataf=np.concatenate(listoffiles)
    return (dataf,labels)

def recordFaces():
    global entry
    global facenumber 
    global names 
    global dataf
    global labels
    camera=cv2.VideoCapture('http://192.168.157.223:8080/video')    
    framecount=0
    data=[]
    while True:
        ret,frame=camera.read()
        if ret:
            grayframe=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            facecoord=face_cascade.detectMultiScale(grayframe,1.3,5)
            for (x,y,width,height) in facecoord:
                face_component=frame[y:y+height,x:x+width,:]
                faceresized=cv2.resize(face_component,(50,50))
                if framecount%10==0 and len(data)<20:
                    data.append(faceresized)
                cv2.rectangle(frame,(x,y),(x+width,y+height),(0,255,255),2)
                info="CAPTURING:"+str((len(data)/20.0)*100)+"%"
                cv2.putText(frame,info,(0,30),font,1,(0,255,255),1)
            framecount+=1
            cv2.imshow('FACE_RECOGNITION_FRAME',frame)
            if cv2.waitKey(1)==27 or len(data)>=20:
                break
        else:
            print("ERROR WHILE OPENING CAMERA")
    camera.release()
    cv2.destroyAllWindows()
    data=np.asarray(data)
    savestr="FACE"+str(facenumber)
    np.save(savestr,data)
    names[facenumber]=entry.get()
    rewriteDictionary()
    names=loadDictionary()
    facenumber=len(names)
    (dataf,labels)=loadLists(names)

        
def distance(x1,x2):
    return np.sqrt(((x1-x2)**2).sum())


def knn(x, train, targets, k=5):
    m = train.shape[0]
    dist = []
    for ix in range(m):
        dist.append(distance(x, train[ix]))
    dist = np.asarray(dist)
    indx = np.argsort(dist)
    sorted_labels = labels[indx][:k]
    counts = np.unique(sorted_labels, return_counts=True)
    return counts[0][np.argmax(counts[1])]


def recogniseFaces():
    '''Detects the faces and recognises the faces present'''
    global listofrecognised
    listofrecognised=[]
    if(len(names)==0):
        return 
    camera=cv2.VideoCapture('http://192.168.157.223:8080/video')    
    while True:
        ret,frame=camera.read()
        inframe=[]
        if ret==True:
            grayframe=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            facecoord=face_cascade.detectMultiScale(grayframe,1.3,5)
            for (x,y,width,height) in facecoord:
                face_component=frame[y:y+height,x:x+width,:]
                faceresized=cv2.resize(face_component,(50,50))
                lab=knn(faceresized.flatten(),dataf,labels)
                Detected_face=names[int(lab)]
                pickle2 = open("detection.pickle","wb")
                pickle.dump(Detected_face, pickle2)
                pickle2.close()
               # gc.collect()
                if Detected_face not in listofrecognised:
                        listofrecognised.append(Detected_face)
                if Detected_face not in inframe:
                    cv2.putText(frame,Detected_face,(x,y),font,1,(255,143,121),2)
                    cv2.rectangle(frame,(x,y),(x+width,y+height),(255,143,121),2)
                    inframe.append(Detected_face)
            cv2.imshow('FACE_RECOGNITION',frame)
            if cv2.waitKey(10)==27:
                break;
        else:
            print("ERROR OPENING CAMERA")
    camera.release()
    cv2.destroyAllWindows()


def saveRecognised():
    global listofrecognised
    f=open('RecognisedList.txt','w')
    f.write('List of people recognised:')
    for names in listofrecognised:
        f.write('\n'+names)
    f.close
    listofrecognised=[]
    

listofrecognised=[]
face_cascade=cv2.CascadeClassifier('./faceCascade.xml')
font=cv2.FONT_HERSHEY_SIMPLEX
names=loadDictionary()
facenumber=len(names)
print(names)
if(facenumber!=0):
    (dataf,labels)=loadLists(names)

root=Tk.Tk()
root.title("FACE DETECTION AND RECOGNITION")
label=Tk.Label(text="ENTER THE NAME IN THE ENTRY FIELD BEFORE RECORDING:")
entry=Tk.Entry(root)
record_button=Tk.Button(text="Record Faces",fg="red",command=recordFaces)
detect_button=Tk.Button(text="Recognise Faces",fg="green",command=recogniseFaces)
clear_button=Tk.Button(text="Clear Records",fg="purple",command=clearRecords)
list_button=Tk.Button(text="Save Recognised",fg="blue",command=saveRecognised)
label.grid(row=0)
entry.grid(row=1,columnspan=100)
record_button.grid(row=2,columnspan=100)
detect_button.grid(row=3,columnspan=100)
clear_button.grid(row=4,columnspan=100)
list_button.grid(row=5,columnspan=100)
root.mainloop()