import dill
import tweepy
import time
import traceback
from ssl import SSLError
from requests.exceptions import Timeout, ConnectionError
from urllib3.exceptions import ReadTimeoutError
from enum import Enum


class ScrapeMode(Enum):
    USERS = 'users'
    TWEETS = 'tweets'


def parse_auth_details(auth_details_file):
    with open(auth_details_file, 'r') as f:
        content = f.readlines()
    auth_details = []
    for i in range(0, len(content), 4):
        auth_detail_i = { t.split(':')[0]: t.split(':')[1].strip() for t in content[i:i+4] }
        auth_details.append(auth_detail_i)
    return auth_details

def get_authentications(auth_details):
    auths = []
    for item in auth_details:
        consumer_key = item['CONSUMER_KEY']
        consumer_secret = item['CONSUMER_SECRET']
        access_key = item['ACCESS_TOKEN']
        access_secret = item['ACCESS_TOKEN_SECRET']

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_key, access_secret)
        auths.append(auth)
    return auths

def limit_handler(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            print("Limit exceeded!")
            time.sleep(15 * 60)
        except (Timeout, SSLError, ReadTimeoutError, ConnectionError) as e:
            print(f"Network error occurred. {str(e)}")
            time.sleep(1)
        except StopIteration:
            break
        except Exception:
            print(traceback.format_exc())
            exit(0)

def save_dill(obj, path):
    with open(path, 'wb+') as f:
        dill.dump(obj, f)

def load_dill(path):
    with open(path, 'rb') as f:
        obj = dill.load(f)

    return obj
