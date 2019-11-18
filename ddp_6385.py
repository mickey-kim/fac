from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import redis
import time
import json
import base64
import io
from PIL import ImageFont, ImageDraw, Image
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

r = redis.StrictRedis(host='58.122.86.255',port=6385)
# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]

max_happy = 0
max_sad = 0
max_neutral = 0
max_angry = 0
max_surprised = 0
max_scared = 0
max_disgust = 0

#feelings_faces = []
#for index, emotion in enumerate(EMOTIONS):
   # feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))

# starting video streaming
#cv2.namedWindow('your_face')
#camera = cv2.VideoCapture(0)
while True:
    try:
        rkey = sorted(r.keys('*'))[0]
        fra = r.get(rkey)
        fra = json.loads(fra)
        id_ = fra['id']
        frame = fra['img']
        r.delete(rkey)
        frame = base64.b64decode(str(frame))
        frame = Image.open(io.BytesIO(frame))
        frame = np.array(frame)
        #reading the frame
        frame = imutils.resize(frame,width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        #canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = frame.copy()
        print('happy one',max_happy,'sad one',max_sad,'neutral one',max_neutral,'angry one',max_angry,'surprised one',max_surprised)
        #time.sleep(0.5)
        if len(faces) > 0:
            label_list = []
            fX_list = []
            fY_list = []
            fW_list = []
            fH_list = []
            prob_list = []
            j_list = []
            for fac in faces:
                #faces = sorted(faces, reverse=True,
                #key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (fX, fY, fW, fH) = faces
                fX_list.append(fX)
                fY_list.append(fY)
                fW_list.append(fW)
                fH_list.append(fH)
                # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
                # the ROI for classification via the CNN
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                
                preds = emotion_classifier.predict(roi)[0]
                emotion_probability = np.max(preds)
                prob_list.append(emotion_probability)
                label = EMOTIONS[preds.argmax()]
                label_list.append(label)
        else: continue

        for i in range(len(label_list)):
            cv2.putText(frameClone, label_list[i], (int(fX_list[i]), int(fY_list[i]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX_list[i], fY_list[i]), (fX_list[i] + fW_list[i], fY_list[i] + fH_list[i]),
            (0, 0, 255), 2)
            r_imgByteArr = io.BytesIO()
            r_imgByteArr.flush()
            img = Image.fromarray(frameClone,'RGB')
            img.save(r_imgByteArr, format='JPEG')
            r_img_read = r_imgByteArr.getvalue()
            r_imgByteArr.flush()
            r_image_64_encode = base64.b64encode(r_img_read)
            r_b64_numpy_arr = np.array(r_image_64_encode)
            emotion = label_list[i]
            send_id = 'result'+id_
            j_list.append([['label',emotion],['prob',prob_list[i]],['img',str(r_b64_numpy_arr)]])
            r.set(send_id, json.dumps(j_list))
            if emotion == 'happy':
                if max_happy <= prob:
                    max_happy = prob
                    r.set('happy mask',str(r_b64_numpy_arr))
            elif emotion == 'neutral':
                if max_neutral <= prob:
                    max_neutral = prob
                    r.set('neutral mask',str(r_b64_numpy_arr))
            elif emotion == 'sad':
                if max_sad <= prob:
                    max_sad = prob
                    r.set('sad mask',str(r_b64_numpy_arr))
            elif emotion == 'angry':
                if max_angry <= prob:
                    max_angry = prob
                    r.set('angry mask',str(r_b64_numpy_arr))
            elif emotion == 'surprised':
                if max_surprised <= prob:
                    max_surprised = prob
                    r.set('surprised mask',str(r_b64_numpy_arr))
            elif emotion == 'disgust':
                if max_angry <= prob:
                    max_angry = prob
                    r.set('disgust mask',str(r_b64_numpy_arr))
            elif emotion == 'scared':
                if max_surprised <= prob:
                    max_surprised = prob
                    r.set('scared mask',str(r_b64_numpy_arr))
    except:
        time.sleep(0.1)
        pass
        ''' 
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            # construct the label text
            text = "{}: {:.2f}%".format(emotion, prob * 100)
    
            # draw the label + probability bar on the canvas
            # emoji_face = feelings_faces[np.argmax(preds)]

                
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5),
            (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 255, 255), 2)
            cv2.putText(frameClone, label, (fX, fY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                          (0, 0, 255), 2)
            r_imgByteArr = io.BytesIO()
            r_imgByteArr.flush()
            img = Image.fromarray(frameClone,'RGB')
            img.save(r_imgByteArr, format='JPEG')
            r_img_read = r_imgByteArr.getvalue()
            r_imgByteArr.flush()
            r_image_64_encode = base64.b64encode(r_img_read)
            r_b64_numpy_arr = np.array(r_image_64_encode)

            #
            #print (label, emotion_probability)
            if emotion == 'happy':
                if max_happy <= prob:
                    max_happy = prob
                    r.set('happy mask',str(r_b64_numpy_arr))
            elif emotion == 'neutral':
                if max_neutral <= prob:
                    max_neutral = prob
                    r.set('neutral mask',str(r_b64_numpy_arr))
            elif emotion == 'sad':
                if max_sad <= prob:
                    max_sad = prob
                    r.set('sad mask',str(r_b64_numpy_arr))
            elif emotion == 'angry':
                if max_angry <= prob:
                    max_angry = prob
                    r.set('angry mask',str(r_b64_numpy_arr))
            elif emotion == 'surprised':
                if max_surprised <= prob:
                    max_surprised = prob
                    r.set('surprised mask',str(r_b64_numpy_arr))
            elif emotion == 'disgust':
                if max_angry <= prob:
                    max_angry = prob
                    r.set('disgust mask',str(r_b64_numpy_arr))
            elif emotion == 'scared':
                if max_surprised <= prob:
                    max_surprised = prob
                    r.set('scared mask',str(r_b64_numpy_arr))
#    for c in range(0, 3):
#        frame[200:320, 10:130, c] = emoji_face[:, :, c] * \
#        (emoji_face[:, :, 3] / 255.0) + frame[200:320,
#        10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)


        cv2.imshow('your_face', frameClone)
        cv2.imshow("Probabilities", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        '''
camera.release()
cv2.destroyAllWindows()
