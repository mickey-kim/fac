#-*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
#merge.py

import redis
import cv2
import io
import os
import base64
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import time
import json

r_label = redis.StrictRedis(port=6380)
r_age = redis.StrictRedis(port=6381)
r_mer = redis.StrictRedis(port=6382,host='58.122.86.230')
r_emo = redis.StrictRedis(port=6383)
while True:
    if not r_label.keys('*'):
        pass
    elif not r_age.keys('*'):
        pass
    elif not r_emo.keys('*'):
        pass
    else:
        try:
            label = str(r_label.get('label'))
            age = int(r_age.get('age'))
            emotion = str(r_emo.get('emotion'))
            if 10<= age < 20:
                age = '10s'
            elif 20<= age < 25:
                age = '20~25'
            elif 25<= age < 30:
                age = '25~30'
            elif 30<= age < 35:
                age = '30~35'
            elif 35<= age < 40:
                age = '35~40'
            elif 40<= age < 50:
                age = '40s'
            elif age < 10:
                age = 'under 9'
            else:
                age = 'over 50'
            gender = str(r_age.get('gender'))
            print ('label : ', label, 'age : ', age, 'gender : ', gender, 'emotion : ', emotion)
            received_img = r_label.get('img_result')
            json_list = []
            json_list.append([['label',label],['age',age],['gender',gender],['emotion',emotion],['img',received_img]])
            r_mer.set('result',json.dumps(json_list))
            #r_mer.hmset('result',{'label':label,'age':age,'gender':gender,'emotion':emotion,'img':received_img})
            decode_img = base64.b64decode(str(received_img))
            decode_img = Image.open(io.BytesIO(decode_img))
            disp_img = np.array(decode_img)
            disp_img = cv2.resize(disp_img,(1280,720))
            cv2.imshow('result_frame',disp_img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        except:
            time.sleep(0.1)
            pass
'''
while True:
    if not sorted(r.keys('*')):
        pass
    else:
        received_img = r.get('img')
        decode_img = base64.b64decode(str(received_img))
        decode_img = Image.open(io.BytesIO(decode_img))
        disp_img = np.array(decode_img)
        disp_img = cv2.resize(disp_img,(1280,720))
        cv2.imshow('frame',disp_img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
'''
