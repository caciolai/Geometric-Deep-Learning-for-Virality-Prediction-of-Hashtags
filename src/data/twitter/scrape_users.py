import tweepy
import argparse

# user classes
from .users_scraper import UserScraper, UserScraperState
from .utils import get_authentications, parse_auth_details


parser = argparse.ArgumentParser(description='Scrape Twitter for users.')
parser.add_argument('--restart', type=bool,
                    help='whether to restart the scraping process')
args = parser.parse_args()

USER_SCRAPING_FOLD = './user_scraping_data'
AUTH_DETAILS_FILE = 'config.txt'

# whether to scrape users or tweets
COLD_START = args.restart

MAX_USERS = 1000000
MAX_CONNECTIONS = 1000
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

    scrape_users(apis)

    print("Done.")


def scrape_users(apis):
    if COLD_START:
        print("Cold start. Looking for origin user...")
        origin_user = UserScraper.get_origin_user(apis[1], max_connections=MAX_CONNECTIONS)
        scraper_state = UserScraperState(first_state=True, origin_user=origin_user)
    else:
        print("Loading scraper state...")
        scraper_state = UserScraperState(first_state=False, data_path=USER_SCRAPING_FOLD)

    scraper = UserScraper(
        data_path=USER_SCRAPING_FOLD,
        state=scraper_state,
        max_users=MAX_USERS,
        max_connections=MAX_CONNECTIONS,
        save_interval=SAVE_INTERVAL
    )

    print("Done.")

    scraper.scrape(apis)


if __name__ == '__main__':
    main()

