
import cv2
import numpy as np
from tkinter import * 
from PIL import ImageTk,Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model("C:/Users/Admin/anaconda3/mask Detection/New_Trained_MaskZ.h5")

video_capture = cv2.VideoCapture(0)

def detection():
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
        faces_list=[]
        preds=[]
        for (x, y, w, h) in faces:
            face_frame = frame[y:y+h,x:x+w]
            face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            face_frame = cv2.resize(face_frame, (224, 224))
            face_frame = img_to_array(face_frame)
            face_frame = np.expand_dims(face_frame, axis=0)
            face_frame =  preprocess_input(face_frame)
            faces_list.append(face_frame)
            if len(faces_list)>0:
                preds = model.predict(faces_list)
            for pred in preds:
                    (mask, withoutMask, notproper) = pred
            if (mask > withoutMask and mask>notproper):
                    label = "Without Mask"
            elif ( withoutMask > notproper and withoutMask > mask):
                    label = "Mask"
            else:
                label = "Wear Mask Properly"

            if label == "Mask":
                color = (0, 255, 0)
            elif label=="Without Mask":
                color = (0, 0, 255)
            else:
                color = (255, 140, 0)
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask, notproper) * 100)
            cv2.putText(frame, label, (x, y- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h),color, 2)
        # Display the resulting frame
        windowname = 'Mask Detection Screen'
        cv2.imshow(windowname, frame)
        keycode = cv2.waitKey(1)
        if cv2.getWindowProperty(windowname, cv2.WND_PROP_VISIBLE) <1:
            break
    video_capture.release()
    cv2.destroyAllWindows()


    
root = Tk()
root.title("Mask Detection System")
root.geometry("700x600")
root.configure(bg="black")
Label(root, text="Face Mask Detection System", font=("times new roman",30,"bold"), bg="black", fg="cyan").pack()
image1 = Image.open("AIMaskE.png")
test = ImageTk.PhotoImage(image1)
label1 = Label(image=test, bg="black")
label1.image = test
label1.pack()
Button(root, text="Start Detection", font=("times new roman", 20, "bold"), bg="grey", fg="cyan", command = detection).pack()

while True:
    
    root.update()