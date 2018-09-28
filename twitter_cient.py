import yaml
import json
from pathlib import Path
from requests_oauthlib import OAuth1Session

BASE_URL = "https://api.twitter.com/"


class TwitterClient:
    __slots__ = ['_client']

    def __init__(self):
        config = self._load_yaml('path/to/yaml.yml')
        self._client = OAuth1Session(
            config.CONSUMER_KEY, config.CONSUMER_SECRET, config.ACCESS_TOKEN, config.ACCESS_TOKEN_SECRET)

    def get_timeline(self, count):
        param = {'count': count}
        url = Path(BASE_URL).joinpath('')
        res = self._client.get()

    def _load_yaml(self, path):
        f = open(path, 'r+')
        data = yaml.load(f)
        return data
