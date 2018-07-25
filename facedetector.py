import os
import sys
import cv2
import numpy
from numpy import ndarray
import math
from typing import List
from logging import getLogger

DEFAULT = 'haarcascades/haarcascade_frontalface_default.xml'
ALT = 'haarcascades/haarcascade_frontalface_alt.xml'
ALT2 = 'haarcascades/haarcascade_frontalface_alt2.xml'
TREE = 'haarcascades/haarcascade_frontalface_alt_tree.xml'

MAX_ROWS = 400
MAX_COLS = 400

class FaceDetector:
    __slots__ = ['_face_cascade', '_eye_cascade', '_logger']
    
    def __init__(self):
        self._face_cascade = cv2.CascadeClassifier(DEFAULT)
        self._eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
        self._logger = getLogger('facedetector')

    def detect_multi(self, img_list: List[ndarray]) -> List[ndarray]:
        return [self.detect(img) for img in img_list]

    def save_and_detect(self, img_list: List[ndarray], save_dir: str) -> None:
        num = 1
        for img in img_list:
            detected_img = self.detect(img)
            cv2.imwrite(os.path.join(save_dir, '{:03}.png'.format(num)), img)
            num += 1

    def resize_image(self, img: ndarray):
        rows, cols, _ = img.shape
        if rows <= MAX_ROWS and cols <= MAX_COLS:
            return img
        if rows > cols:
            resize_rows = MAX_ROWS
            resize_cols = int(cols * MAX_ROWS / rows)
        else:
            resize_rows = int(rows * MAX_COLS / cols)
            resize_cols = MAX_COLS
        resize_img = cv2.resize(img, (resize_cols, resize_rows))
        return resize_img
            
    def rotate_image(self, img: ndarray, save_dir, num):
        resize_img = self.resize_image(img)
        rows, cols, colors = resize_img.shape
        hypot = int(math.hypot(rows, cols))
        print(rows, cols, colors, hypot)
        frame = numpy.zeros((hypot, hypot, colors), numpy.uint8)
        frame[int((hypot - rows) * 0.5):int((hypot + rows) * 0.5), int((hypot - cols) * 0.5):int((hypot + cols) * 0.5)] = resize_img
     
        for deg in range(-50, 51, 5):
            M = cv2.getRotationMatrix2D((hypot * 0.5, hypot * 0.5), -deg, 1.0)
            rotated = cv2.warpAffine(frame, M, (hypot, hypot))
            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            faces = self._face_cascade.detectMultiScale(gray)
            for (x, y, w, h) in faces:
                cv2.rectangle(rotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                eyes = self._eye_cascade.detectMultiScale(rotated)
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(rotated,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.imwrite(os.path.join(save_dir, '{:03}_{}.png'.format(num, deg)), rotated)
            
    def save_and_triming(self, img_list: List[ndarray], save_dir: str) -> None:
        triming_list = []
        for img in img_list:
            triming_list.extend(self.triming(img))
        
        num = 1
        for img in triming_list:
            cv2.imwrite(os.path.join(save_dir, '{:03}.png'.format(num)), img)
            num += 1
            
    def triming(self, img: ndarray) -> List[ndarray]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray)
        image_list = []
        for (x,y,w,h) in faces:
            # eyes = self._eye_cascade.detectMultiScale(gray)
            # if not len(eyes) is 2:
                # continue
            if not x:
                continue
            triming_image = img[y:y+h, x:x+w]
            gray_triming = cv2.cvtColor(triming_image, cv2.COLOR_BGR2GRAY)
            eyes = self._eye_cascade.detectMultiScale(gray_triming)
            # for (ex,ey,ew,eh) in eyes:
                # cv2.rectangle(triming_image,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            if len(eyes) > 1:
                image_list.append(triming_image)
        return image_list

    def detect(self, img: ndarray) -> ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 顔を検知
        faces = self._face_cascade.detectMultiScale(gray)
        for (x,y,w,h) in faces:
            # 検知した顔を矩形で囲む
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            # 顔画像（グレースケール）
            roi_gray = gray[y:y+h, x:x+w]
            # 顔ｇ増（カラースケール）
            roi_color = img[y:y+h, x:x+w]
            # 顔の中から目を検知
            eyes = self._eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                # 検知した目を矩形で囲む
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        return img

    def _is_image(self, image_file: str) -> bool:
        filename, ext = os.path.splitext(image_file)
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            return True
        else:
            return False

    def load_image(self, image_file: str) -> ndarray:
        self._logger.info('load: {}'.format(image_file))
        if os.path.isfile(image_file) and self._is_image(image_file):
            return cv2.imread(image_file)
        
    def load_images(self, image_dir: str) -> List[ndarray]:
        files = os.listdir(image_dir)
        image_list = []
        for file_name in files:
            self._logger.info('load: {}'.format(file_name))
            if os.path.isfile(os.path.join(image_dir, file_name)) and self._is_image(file_name):
                image_list.append(cv2.imread(os.path.join(image_dir, file_name)))
        return image_list
    
    def save_image(self, img_list: List[ndarray], save_dir: str) -> None:
        self._logger('save dir: {}'.format(save_dir))
        num = 1
        for img in img_list:
            cv2.imwrite(os.path.join(save_dir, '{:03}.png'.format(num)), img)
            num += 1
    
    def run(self, image_dir: str, save_dir: str) -> None:
        # img_list = self.load_images(image_dir)
        # self.save_and_triming(img_list, save_dir)
        # self.save_and_detect(img_list, save_dir)
        # detected_img_list = self.detect_multi(img_list)
        # self.save_image(detected_img_list, save_dir)
        n = 1
        img = self.load_image(image_dir)
        self.rotate_image(img, save_dir, n)
        '''
        for img in img_list:
            self.rotate_image(img, save_dir, n)
            n += 1
        '''


if __name__ == '__main__':
    argv = sys.argv
    detector = FaceDetector()
    save_dir = argv[2] or 'detected/'
    detector.run(argv[1], save_dir)
