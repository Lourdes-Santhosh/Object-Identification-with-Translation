import numpy as np
import time
import cv2
import pyttsx3
from gtts import gTTS
from deep_translator import GoogleTranslator
import pygame


LABELS = open("coco.names").read().strip().split("\n")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
font = cv2.FONT_HERSHEY_PLAIN

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# initialize
cap = cv2.VideoCapture(0)
# initialize the audio using pyttsx3
audio = pyttsx3.init()

frame_count = 0
start = time.time()
first = True
frames = []
flag = 1

while True:

    frame_count += 1
    
    ret, frame = cap.read()
    cv2.imshow("Video", frame)
    frames.append(frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    if ret:
        key = cv2.waitKey(1)
        if frame_count % 60 == 0:  # To make delay between every 60 - Frames
            end = time.time()
            # grab the frame dimensions and convert it to a blob
            (H, W) = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)

            # initialize our lists of detected bounding boxes, confidences, and
            # class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []
            centers = []

            for output in layerOutputs:
               
                for detection in output:
                    
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    if confidence > 0.5:

                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                        centers.append((centerX, centerY))

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

            for i in range(len(boxes)):
                if i in idxs:
                    x, y, w, h = boxes[i]
                    label = str(classes[classIDs[i]])
                    confidence = confidences[i]
                    color = colors[classIDs[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(
                        frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)

            texts = ["The Objects infront of you are "]

            # ensure at least one detection exists
            if len(idxs) > 0:
               
                for i in idxs.flatten():
                    # find positions
                    centerX, centerY = centers[i][0], centers[i][1]

                    if centerX <= W/3:
                        W_pos = "left "
                    elif centerX <= (W/3 * 2):
                        W_pos = "center "
                    else:
                        W_pos = "right "

                    if centerY <= H/3:
                        H_pos = "top "
                    elif centerY <= (H/3 * 2):
                        H_pos = "mid "
                    else:
                        H_pos = "bottom "

                    texts.append(H_pos + W_pos + LABELS[classIDs[i]])
                    # print(H_pos + W_pos + LABELS[classIDs[i]])
                    if H_pos + W_pos + LABELS[classIDs[i]] != ' ':
                        flag = 0
                    else:
                        flag = 1

            print(texts)
            exe_char = input("Enter (z - Sound / t - Translation) : ")
            voices = audio.getProperty('voices')

            audio.setProperty('voice', voices[1].id) # changing index, changes voices. 0 for male
            # For SOUND OUTPUT
            if exe_char == 'z':
                if (flag == 0):

                    description = ', '.join(texts)
                    print(description)
                else:
                    texts = "There are no Objects infront of you , sorry"
                    print(texts)
                audio.say(texts)
                audio.runAndWait()

            # For TRANSLATION
            if exe_char == 't':
                translated_to = "ta"
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                else:
                    if flag == 0:
                        description = ', '.join(texts)
                        print(description)
                        translated_text = GoogleTranslator(
                            source='auto', target=translated_to).translate(description)
                    else:
                        translated_text = GoogleTranslator(source='auto', target=translated_to).translate(
                            "There are no Objects infront of you , sorry")

                    tts = gTTS(text=translated_text, lang=translated_to, slow=False)
                    tts.save("translated_audio.mp3")

                    try:
                        pygame.mixer.init()
                        sound = pygame.mixer.Sound("translated_audio.mp3")
                        sound.play()
                        pygame.time.delay(int(sound.get_length() * 1000))

                    except Exception as err:

                        print("An error occurred:", type(err).__name__)

cap.release()
cv2.destroyAllWindows()
