#-*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
# face_recog.py
import redis
import face_recognition
import cv2
import csv
#import camera
import os
import io
import numpy as np
import base64
from PIL import ImageFont, ImageDraw, Image
import time
r = redis.StrictRedis()
r2 = redis.StrictRedis(port=6380)
class FaceRecog():
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        #self.camera = camera.VideoCamera()
        self.known_face_encodings = []
        self.known_face_names = []

        # Load sample pictures and learn how to recognize it.
        dirname = 'knowns'
        global files
        files = os.listdir(dirname)
        #print files, type(files)
        #start_time = time.time()
        for filename in files:
            name, ext = os.path.splitext(filename)
            #name = name.encode("utf-8")
            #print type(name)
            if ext == '.jpg':
                self.known_face_names.append(name)
                pathname = os.path.join(dirname, filename)
                img = face_recognition.load_image_file(pathname)
                face_encoding = face_recognition.face_encodings(img)[0]
                #print face_encoding, type(face_encoding)
                start_time = time.time()
                r.set(name,base64.encodestring(face_encoding.tobytes()).decode('ascii'))
                ddd = r.get(name)
                ddd = np.fromstring(base64.decodestring(bytes(ddd.decode('ascii'))), dtype='float64')
                print ddd, type(ddd)
                self.known_face_encodings.append(ddd)
                #print(time.time()-start_time)
        #print(time.time()-start_time)
        # for loop -> 지점별 폴더이름마다 a_face_locations=[]등을 생성
        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

    #def __del__(self):
        #del self.camera

    
    def get_frame(self):
        # Grab a single frame of video
        #frame = self.camera.get_frame()
        #r = redis.StrictRedis()
        frame = r2.get('raw_img')
        r2.delete('raw_img') #added ###
        frame = base64.b64decode(str(frame))
        frame = Image.open(io.BytesIO(frame))
        frame = np.array(frame)
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if self.process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                # See if the face is a match for the known face(s)
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                min_value = min(distances)
                print(distances)
                # tolerance: How much distance between faces to consider it a match. Lower is more strict.
                # 0.6 is typical best performance.
                name = "Unknown"
                global flag
                flag = False
                if min_value < 0.4:
                    index = np.argmin(distances)
                    name = self.known_face_names[index].encode("UTF-8")
                else:
                    i = len(files) + 1
                    cv2.imwrite("./knowns/"+str(i)+".jpg",frame)
                    flag = True
                r2.set('label', name)
                self.face_names.append(name)
                visit = csv.reader(open('visit.csv','rb'),delimiter=',')
                visit_row = []
                if flag == True:
                    name = i
                print ('new name', name)
                for row in visit:
                    if name == row[0]:
                        #print row
                        row[1] = int(row[1])+1
                    visit_row.append(row)
                if flag == True:
                    visit_row.append([name, 1])
                #visit.close()
                print visit_row
                with open('visit.csv','wb') as vt:
                    writer = csv.writer(vt)
                    for row in visit_row:
                        writer.writerow(row)

        self.process_this_frame = not self.process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 1)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return frame

    def get_jpg_bytes(self):
        frame = self.get_frame()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()

import time
if __name__ == '__main__':
    face_recog = FaceRecog()
    #print(face_recog.known_face_names)
    while True:
        try:
            frame = face_recog.get_frame()
            r_imgByteArr = io.BytesIO()
            r_imgByteArr.flush()
            img = Image.fromarray(frame,'RGB')
            img.save(r_imgByteArr, format='JPEG')
            r_img_read = r_imgByteArr.getvalue()
            r_imgByteArr.flush()
            r_image_64_encode = base64.b64encode(r_img_read)
            r_b64_numpy_arr = np.array(r_image_64_encode)
            #r_json = json.dumps(r_b64_numpy_arr)
            r2.set('img_result',str(r_b64_numpy_arr))
            if flag == True:
                print("argv was",sys.argv)
                print("sys.executable was", sys.executable)
                print("restart now")
                os.execv(sys.executable, ['python'] + sys.argv)
                #print('uploading')
            # show the frame
            #cv2.imshow("Frame", frame)
            #key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            #if key == ord("q"):
            #    break
        except:
            time.sleep(0.1)
            pass

    # do a bit of cleanup
    #cv2.destroyAllWindows()
    #print('finish')
