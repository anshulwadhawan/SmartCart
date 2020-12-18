The intelligent cart prepares an invoice of fruits present in the cart after detection, for the user recognised by face.
Implemented Single Shot Detector (SSD) Model for object detection (among fruits) and KNN Algorithm for facial recognition difference evaluations.

Ensure you are using Python 2.7 with the following installed:
1.opencv
2.tkinter
3.pickle
4.numpy

Running the Program:
Ensure that the files "haarcascade_ff" and "faces.pickle" are in the same directory.
Run "userCamera.py" and an interface with buttons should open.
If the program doesn't work, click ResetandDebug which runs a script to reset the dictionary and clear "faces.pickle".

Using the Program:
If there are no faces recorded, the recognise button does nothing.
Type the name of the person whose face is being recorded and click the record faces button.
After all people who have to be recognised are recorded you can click the recognise faces button and it should detect faces.
The clear faces button clears the saved records.
The save faces button saves the faces recognised into a text file in the same directory.
