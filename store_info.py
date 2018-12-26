# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import redis
import pickle
import time
import os
import numpy as np
from sys import platform

# Remember to add your installation path here
# Option a
# Option b
# If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
# sys.path.append('/usr/local/python')
dir_path = os.path.dirname(os.path.realpath(__file__))

if platform == "win32":
    sys.path.append(dir_path + '/../../python/openpose/')
else:
    sys.path.append('../../python')
try:
    from openpose import *
except:
    raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')

redis = redis.Redis(host='localhost', port=6379, db=0)

for elem in redis.keys():
    redis.delete(elem)

# Parameters for OpenPose. Take a look at C++ OpenPose example for meaning of components. Ensure all below are filled

def pross():
    params = dict()
    params["logging_level"] = 3
    params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x368"
    params["model_pose"] = "BODY_25"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.3
    params["scale_number"] = 1
    params["render_threshold"] = 0.05
    # If GPU version is built, and multiple GPUs are available, set the ID here
    params["num_gpu_start"] = 0
    params["disable_blending"] = False
    # Ensure you point to the correct path where models are located
    params["default_model_folder"] = dir_path + "/../../../models/"
    # Construct OpenPose object allocates GPU memory
    openpose = OpenPose(params)

    #video_path = "/woody/software/source/openpose/examples/media/video.avi"
    video_path = "/home/woody/tmp/openpose/test.mp4"
    video = cv2.VideoCapture()

    if not video.open(video_path):
       print("can not open the video")
       exit(1)
    count = 1
    index = 1
    start = time.time()
    while True:
       _, frame = video.read()
       if frame is None:
          break
       if count % 15 == 0:
          keypoints,scores,output_image = openpose.forward(frame, True)
          img_encode = cv2.imencode('.jpg', output_image)[1]
          data_encode = np.array(img_encode)
          str_encode = data_encode.tostring()
          data = {}
          # data["image"] = pickle.dumps(output_image)
          data["image"] = str_encode
          data["keypoints"] = keypoints
          data["scores"] = scores
          try:
            pipe = redis.pipeline()
            pipe.multi()
            redis.lpush('keysList', "image_info_" + str(index))
            redis.set("image_info_" + str(index), pickle.dumps(data))
            # pipe.set("keypoints_info_" + str(index),pickle.dumps(keypoints))
            pipe.execute()
          except Exception as err:
            print(err)
          # Print the human pose keypoints, i.e., a [#people x #keypoints x 3]-dimensional numpy object with the keypoints of all the people on that image
          # print(keypoints)
          # save_path = "{}/{:>03d}.jpg".format("/home/woody/tmp/openpose", index)
          # cv2.imwrite(save_path, frame)
          index += 1
       count += 1
    print(">>>" + str(time.time() - start))
    video.release()
    print("Totally save {:d} pics".format(index - 1))

    ret = redis.lrange("keysList", 0, 1)
    for key in ret:
       ret_data = redis.get(key)
       if ret_data is None:
        continue
       ret = pickle.loads(ret_data)
       # print("keypoints:" + str(ret["keypoints"]))
       print("scores:" + str(ret["scores"]))
       image = np.asarray(bytearray(ret["image"]), dtype="uint8")
       image = cv2.imdecode(image, cv2.IMREAD_COLOR)
       cv2.imshow('test_tset', image)
       # cv2.imshow('test_tset',pickle.loads(ret["image"]))
       cv2.waitKey(0)


if __name__ == '__main__':
    pross()
