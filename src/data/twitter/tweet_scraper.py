import tweepy
from tweet import Tweet
from utils import save_dill, load_dill
import os


class TweetScraperState:
    def __init__(
            self,
            tweets=None,
            users_queue=None,
            time_window=None,
    ):
        """
        :param tweets: Tweets scraped so far
        :param users_queue: Queue of users to be processed
        :param time_window: Time window of reference for the tweets
        """
        self.tweets = tweets
        self.users_queue = users_queue
        self.time_window = time_window

    @staticmethod
    def load(folder):
        tweets_path = os.path.join(folder, "tweets.pkl")
        users_queue_path = os.path.join(folder, "users_queue.pkl")
        time_window_path = os.path.join(folder, "time_window.pkl")

        tweets = []
        users_queue = []
        time_window = None

        # load users
        try:
            tweets = load_dill(tweets_path)
            users_queue = load_dill(users_queue_path)
            time_window = load_dill(time_window_path)
        except Exception as exc:
            print("ERROR ON DATA LOADING!")
            print(exc)
            exit(-1)

        return TweetScraperState(tweets=tweets, users_queue=users_queue, time_window=time_window)

    def save(self, folder, iteration):
        print("Saving scraper state...")
        tweets_path = os.path.join(folder, f"tweets_{iteration}.pkl")
        users_queue_path = os.path.join(folder, "users_queue.pkl")
        time_window_path = os.path.join(folder, "time_window.pkl")

        save_dill(self.tweets, tweets_path)
        save_dill(self.users_queue, users_queue_path)
        save_dill(self.time_window, time_window_path)
        print('Done.')


class TweetScraper:
    def __init__(
            self,
            data_path,
            state=None,
            save_interval=10,
    ):
        self.data_path = data_path
        self.state = state
        self.save_interval = save_interval

    def scrape(self, apis):
        assert not (self.data_path is None or self.state is None)
        assert not (self.state.tweets is None or self.state.users_queue is None or self.state.time_window is None)

        api_idx = 0

        tweets, users_id_queue, time_window = self.state.tweets, self.state.users_queue, self.state.time_window
        ts, te = time_window

        scraped_users = {key for key, value in tweets.items()}
        to_scrape_queue = [ user for user in users_id_queue if user not in scraped_users ]

        try:
            iterations = 0
            tweets_count = sum([len(v) for k, v in tweets.items()])
            while len(to_scrape_queue) > 0:
                user_id = to_scrape_queue.pop(0)

                print(f"\n\nUsers left: {len(to_scrape_queue)}.\nTweets so far: {tweets_count}.\n")
                api = apis[api_idx]

                # Getting tweets
                user_tweets = []
                try:
                    for tweet_page in tweepy.Cursor(api.user_timeline, user_id=user_id, tweet_mode="extended").pages():
                        user_tweets.extend(tweet_page)
                        last_timestamp = user_tweets[-1].created_at

                        if last_timestamp < ts:
                            break
                except tweepy.error.TweepError as exc:
                    print(f"\nCatched TweepError ({exc})."
                          f"\nIgnoring user.")
                    continue

                # Retaining only tweets in [t_s, t_e]
                user_tweets_list = []
                for tweet in user_tweets:
                    if ts <= tweet.created_at <= te:
                        tweet = Tweet(tweet)
                        assert tweet.author == user_id
                        user_tweets_list.append(tweet)

                # Register user tweets
                tweets[user_id] = user_tweets_list
                tweets_count += len(user_tweets_list)

                # Switch api to balance load
                api_idx = (api_idx + 1) % 2

                # Save progress
                iterations += 1
                if iterations % self.save_interval == 0:
                    self.state.save(self.data_path, iterations)

        except KeyboardInterrupt:
            print("\n\nInterrupt received. Terminating...")
        finally:
            self.state.save(self.data_path, self.save_interval)
