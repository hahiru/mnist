import os
import cv2
import yaml
import json
from requests_oauthlib import OAuth1Session
import urllib

BASE_URL = "https://api.twitter.com/"


class TwitterClient:
    __slots__ = ['_client', '_func']

    def __init__(self, func):
        config = self._load_yaml('config/api.yml')
        self._client = OAuth1Session(
            config['CONSUMER_KEY'], config['CONSUMER_SECRET'], config['ACCESS_TOKEN'], config['ACCESS_TOKEN_SECRET'])
        self._func = func

    def get_timeline(self, count=5):
        params = {'count': count}
        url = os.path.join(BASE_URL, '1.1/statuses/user_timeline.json')
        res = self._client.get(url, params=params)

        if res.status_code == 200:
            timelines = json.loads(res.text)
            self._func(timelines)
        else:
            print("Failed: %d" % res.status_code)

    def get_user_timeline(self, screen_name, count=5):
        params = {'screen_name': screen_name, 'count': count}
        url = os.path.join(BASE_URL, '1.1/statuses/user_timeline.json')
        res = self._client.get(url, params=params)

        if res.status_code == 200:
            timelines = json.loads(res.text)
            self._func(timelines)
        else:
            print("Failed: %d" % res.status_code)

    def _load_yaml(self, path):
        f = open(path, 'r+')
        data = yaml.load(f)
        return data


if __name__=='__main__':
    def display_timeline(timelines):
        for line in timelines:
            print(('::').join([line['user']['name'], line['text'], line['created_at']]))
            print('*******************************************')

    def take_images(timelines, dir_name='images/twitter/'):
        for line in timelines:
            screen_name = line['user']['screen_name']
            save_dir = os.path.join(dir_name, screen_name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            if line.get('extended_entities'):
                for media in line['extended_entities']['media']:
                    print(media['media_url'])
                    image_url = media['media_url']
                    with open(os.path.join(save_dir, os.path.basename(image_url)), 'wb') as f:
                        img = urllib.request.urlopen(image_url).read()
                        f.write(img)

    twitter_client = TwitterClient(take_images)
    # twitter_client.get_timeline()
    twitter_client.get_user_timeline('hinata_980115', count=100)
