from datetime import datetime, timedelta
import tweepy
import argparse
import os

# user classes
from .tweet_scraper import TweetScraper, TweetScraperState
from .utils import parse_auth_details, get_authentications, load_dill

parser = argparse.ArgumentParser(description='Scrape Twitter for tweets.')
parser.add_argument('--restart', type=bool,
                    help='whether to restart the scraping process')
args = parser.parse_args()

TWEET_SCRAPING_FOLD = './tweet_scraping_data'
AUTH_DETAILS_FILE = 'config.txt'

# whether to scrape users or tweets
COLD_START = args.restart
TIME_WINDOW_DAYS = 7
SAVE_INTERVAL = 1000


def main():
    print('Handling authentication..')
    auth_details = parse_auth_details(AUTH_DETAILS_FILE)
    auths = get_authentications(auth_details)

    print("Creating API handler(s)...")
    apis = [
        tweepy.API(auths[0], wait_on_rate_limit=True, wait_on_rate_limit_notify=True),
        tweepy.API(auths[1], wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    ]

    scrape_tweets(apis)

    print("Done.")


def scrape_tweets(apis):
    
    users_path = os.path.join(TWEET_SCRAPING_FOLD, "users.pkl")
    users = load_dill(users_path)
    users_queue = list(set([user.id for user in users]))

    if COLD_START:
        print("Cold start. Creating empty state...")
        te = datetime.now()
        ts = te - timedelta(days=TIME_WINDOW_DAYS)
        time_window = (ts, te)

        print(len(users_queue))
        scraper_state = TweetScraperState(
            users_queue=users_queue,
            time_window=time_window,
            tweets=dict()
        )
    else:

        print("Loading scraper state...")
        scraper_state = TweetScraperState.load(TWEET_SCRAPING_FOLD)
        scraper_state.users_queue = users_queue

    scraper = TweetScraper(
        TWEET_SCRAPING_FOLD,
        state=scraper_state,
        save_interval=SAVE_INTERVAL
    )
    print("Done.")

    scraper.scrape(apis)


if __name__ == '__main__':
    main()

