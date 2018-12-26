# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import redis
import pickle
import cmath
import time
import os
import numpy as np
from sys import platform


redis = redis.Redis(host='localhost', port=6379, db=0)

def calcHipAngle(x1,y1,x2,y2,x3,y3):
    if x1 == 0 or y1 == 0 or x2 == 0 or y2 == 0 or x3 == 0 or y3 == 0:
        return 0,True
    a2 = (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)
    b2 = (x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)
    c2 = (x1-x3)*(x1-x3)+(y1-y3)*(y1-y3)
    a = cmath.sqrt(a2)
    b = cmath.sqrt(b2)
    c = cmath.sqrt(c2)
    pos = (a2+b2-c2)/(2*a*b)
    angle = cmath.acos(pos)
    realangle = angle*180/cmath.pi
    print("calcHipAngle:" + str(realangle.real));
    if (realangle.real >= 30 and realangle.real <= 140) :
        return realangle.real,True
    else:
        return realangle.real,False

def calcKneeAngle(x1,y1,x2,y2,x3,y3):
    if x1 == 0 or y1 == 0 or x2 == 0 or y2 == 0 or x3 == 0 or y3 == 0:
        return 0,True
    a2 = (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)
    b2 = (x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)
    c2 = (x1-x3)*(x1-x3)+(y1-y3)*(y1-y3)
    a = cmath.sqrt(a2)
    b = cmath.sqrt(b2)
    c = cmath.sqrt(c2)
    pos = (a2+b2-c2)/(2*a*b)
    angle = cmath.acos(pos)
    realangle = angle*180/cmath.pi
    print("calcKneeAngle:" + str(realangle.real))
    if (realangle.real <= 140) :
        return realangle.real,True
    else:
        return realangle.real,False

def calcLenRate(x1,y1,x2,y2,x3,y3):
    if x1 == 0 or y1 == 0 or x2 == 0 or y2 == 0 or x3 == 0 or y3 == 0:
        return 1,False
    a2 = (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)
    b2 = (x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)
    c2 = (x1-x3)*(x1-x3)+(y1-y3)*(y1-y3)
    d2 = (y1-y2)*(y1-y2)
    a = cmath.sqrt(a2)
    b = cmath.sqrt(b2)
    c = cmath.sqrt(c2)
    d = cmath.sqrt(d2)
    result = max(a.real, b.real)/d.real
    print("calcLenRate:" + str(result))
    if result >= 0.8 and result <= 1.2 :
        return result,False
    else:
        return result,True


def pross():
    count = 0
    imageArray = {}
    start = time.time()
    ret = redis.lrange("keysList", 0, -1)
    tmp_count = 0;
    for key in ret:
       ret_data = redis.get(key)
       if ret_data is None:
        continue
       ret = pickle.loads(ret_data)
       flag = False
       for keypoint in ret["keypoints"]:
           if flag:
               break
           x1 = keypoint[2][0]
           y1 = keypoint[2][1]
           x2 = keypoint[9][0]
           y2 = keypoint[9][1]
           x3 = keypoint[10][0]
           y3 = keypoint[10][1]
           r_result,r_flag = calcHipAngle(x1,y1,x2,y2,x3,y3)
           x1 = keypoint[5][0]
           y1 = keypoint[5][1]
           x2 = keypoint[12][0]
           y2 = keypoint[12][1]
           x3 = keypoint[13][0]
           y3 = keypoint[13][1]
           l_result,l_flag = calcHipAngle(x1,y1,x2,y2,x3,y3)

           x1 = keypoint[9][0]
           y1 = keypoint[9][1]
           x2 = keypoint[10][0]
           y2 = keypoint[10][1]
           x3 = keypoint[11][0]
           y3 = keypoint[11][1]
           ra_result,ra_flag = calcKneeAngle(x1,y1,x2,y2,x3,y3)
           ra_len_result,ra_len_flag = calcLenRate(x1,y1,x2,y2,x3,y3)
           x1 = keypoint[12][0]
           y1 = keypoint[12][1]
           x2 = keypoint[13][0]
           y2 = keypoint[13][1]
           x3 = keypoint[14][0]
           y3 = keypoint[14][1]
           la_result,la_flag= calcKneeAngle(x1,y1,x2,y2,x3,y3)
           la_len_result,la_len_flag = calcLenRate(x1,y1,x2,y2,x3,y3)
           if  ra_flag and la_flag :
               flag = True
           if  ra_len_flag and la_len_flag :
               flag = True
           if (r_flag or l_flag) and (abs(r_result - l_result) <= 30) and (ra_result or la_result):
               flag = True
           if la_result >= 170 or ra_result >= 170:
               flag = False

           if la_result >= 170 or ra_result >= 170:
               flag = False

           if (la_len_result >= 0.9 and la_len_result <= 1.09) or (ra_len_result >= 0.9 and ra_len_result <= 1.09):
               flag = False
               #return True
       if flag:
           image = np.asarray(bytearray(ret["image"]), dtype="uint8")
           image = cv2.imdecode(image, cv2.IMREAD_COLOR)
           save_path = "{}/{:>03s}.jpg".format("/home/woody/tmp/openpose/li", str(key))
           cv2.imwrite(save_path, image)
           if 'index' in imageArray.keys():
               if imageArray['start'] >= count -10:
                   if imageArray['index'] == count -1:
                       imageArray['start'] = count
                       imageArray['count'] = imageArray['count'] + 1
                   imageArray['index'] = count
               else:
                   save_path = "{}/{:>03s}.jpg".format("/home/woody/tmp/openpose/test", str(imageArray['key']))
                   if imageArray['count'] > 1:
                       cv2.imwrite(save_path, imageArray["image"])
                   imageArray = {}
                   count = 0
           else:
               imageArray['index'] = count
               imageArray['start'] = count
               imageArray['count'] = 0
               image_save = np.asarray(bytearray(ret["image"]), dtype="uint8")
               image_save = cv2.imdecode(image_save, cv2.IMREAD_COLOR)
               imageArray['image'] = image_save
               imageArray['key'] = key

       else:
           image = np.asarray(bytearray(ret["image"]), dtype="uint8")
           image = cv2.imdecode(image, cv2.IMREAD_COLOR)
           save_path = "{}/{:>03s}.jpg".format("/home/woody/tmp/openpose/wu", str(key))
           cv2.imwrite(save_path, image)
       count = count + 1
    print(str(count) + ">>>" + str(time.time() - start))

if __name__ == '__main__':
      pross()
