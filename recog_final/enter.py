import cv2
import base64
import redis
import io
import numpy as np
import json
from PIL import ImageFont, ImageDraw, Image

r = redis.StrictRedis()
r_rec = redis.StrictRedis(port=6380)
r_age = redis.StrictRedis(port=6381)
r_emo = redis.StrictRedis(port=6383)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("result", frame)
        k = cv2.waitKey(5)
        if k == ord('s'):
            r_imgByteArr = io.BytesIO()
            r_imgByteArr.flush()
            img = Image.fromarray(frame,'RGB')
            img.save(r_imgByteArr, format='JPEG')
            r_img_read = r_imgByteArr.getvalue()
            r_imgByteArr.flush()
            r_image_64_encode = base64.b64encode(r_img_read)
            r_b64_numpy_arr = np.array(r_image_64_encode)
            #r_json = json.dumps(r_b64_numpy_arr)
            r_rec.set('raw_img',str(r_b64_numpy_arr))
            r_age.set('raw_img',str(r_b64_numpy_arr))
            r_emo.set('raw_img',str(r_b64_numpy_arr))
            print ("send")
        #print ("next")
    else:
        pass
