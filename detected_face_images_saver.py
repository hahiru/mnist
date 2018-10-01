import io
import os
import numpy as np
import requests
import cv2
from twitterclient import TwitterClient
from facedetector import FaceDetector


def save_detected_images(timelines, dir_name='images/detected/alt-neighbors2'):
    facedetector = FaceDetector()
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    for line in timelines:
        screen_name = line['user']['screen_name']
        save_dir = os.path.join(dir_name, screen_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if line.get('extended_entities'):
            for media in line['extended_entities']['media']:
                image_url = media['media_url']
                res = requests.get(image_url)
                bin_data = io.BytesIO(res.content)
                file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)

                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if facedetector.is_face_image(img):
                    print(image_url)
                    cv2.imwrite(os.path.join(save_dir, os.path.basename(image_url)), img)


twitter_client = TwitterClient(save_detected_images)
twitter_client.get_user_timeline('hinata_980115', count=100)
