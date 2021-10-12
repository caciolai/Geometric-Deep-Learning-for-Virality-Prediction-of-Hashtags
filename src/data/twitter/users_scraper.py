import os
import random
import tweepy


from .user import User
from .utils import load_dill, save_dill, limit_handler

class UserScraperState:
    def __init__(self, first_state, origin_user=None, data_path=None):

        assert (first_state and origin_user) or data_path

        if first_state:
            self.queue = [origin_user.id]
            self.visited_ids = set()
            self.users = []
            self.edges = []
        else:
            self.load_state(data_path)

    def load_state(self, folder):

        users_path = os.path.join(folder, "users.pkl")
        edges_path = os.path.join(folder, "edges.pkl")
        queue_path = os.path.join(folder, 'queue.pkl')
        visited_ids_path = os.path.join(folder, 'visited.pkl')

        try:
            self.users = load_dill(users_path)
            self.edges = load_dill(edges_path)
            self.queue = load_dill(queue_path)
            self.visited_ids = set(load_dill(visited_ids_path))
        except Exception as exc:
            print(f'Error on data loading.')
            print(exc)
            exit(0)

    def save(self, folder):
        print("Saving scraper state...")

        users_path = os.path.join(folder, "users.pkl")
        edges_path = os.path.join(folder, "edges.pkl")
        queue_path = os.path.join(folder, 'queue.pkl')
        visited_ids_path = os.path.join(folder, 'visited.pkl')

        save_dill(self.users, users_path)
        save_dill(self.edges, edges_path)
        save_dill(self.queue, queue_path)
        save_dill(list(self.visited_ids), visited_ids_path)

        print("Done.")

    def __repr__(self):
        return f"Queue: {self.queue}\n\n" \
               f"Visited IDs: {self.visited_ids}\n\n" \
               f"Users: {self.users}\n\n" \
               f"Edges: {self.edges}\n\n"

class UserScraper:

    def __init__(
            self,
            data_path,
            state=None,
            max_users=100000,
            max_connections=1000,
            save_interval=10,
    ):
        self.data_path = data_path
        self.state = state
        self.max_users = max_users
        self.max_connections = max_connections
        self.save_interval = save_interval

    # TODO: handle somehow nodes with many connections
    # TODO: save enough_in_queue state to file, otherwise when reloaded it will again add users to the queue

    def scrape(self, apis):
        assert self.data_path is not None and self.state is not None

        queue, visited_ids, users, edges = self.state.queue, self.state.visited_ids, self.state.users, self.state.edges

        queue_set = set(queue)

        followers_api, followees_api = 0, 1
        print("Starting scraping...")
        iterations = 0

        enough_in_queue = False

        try:
            while len(users) < self.max_users and len(queue) > 0:
                api = apis[iterations % 2]

                # print(f'There are {apis[followers_api].rate_limit_status()} requests left for'
                #       f'followers and {apis[followees_api].rate_limit_status()} for followes')

                if len(queue) >= self.max_users:
                    enough_in_queue = True

                print(f'\n\nUsers: {len(users)}.\nEdges: {len(edges)}.\nQueue: {len(queue)}.')
                user_id = queue.pop(0)

                try:
                    user = api.get_user(user_id)
                    user = User(user)
                except tweepy.error.TweepError as exc:
                    print(f"\nCatched TweepError ({exc}). Ignoring user.")
                    continue

                users.append(user)
                visited_ids.add(user.id)

                if user.attrs.friends_count > self.max_connections or user.attrs.followers_count > self.max_connections:
                    continue

                try:
                    follower_cursor = tweepy.Cursor(api.followers_ids, id=user.id).pages()
                    followee_cursor = tweepy.Cursor(api.friends_ids, id=user.id).pages()
                    followers = self.get_connections_list(follower_cursor)
                    followees = self.get_connections_list(followee_cursor)
                except tweepy.error.TweepError as exc:
                    print(f"\nCatched TweepError ({exc}). Ignoring user.")
                    continue

                for follower_id in followers:
                    if follower_id not in visited_ids:
                        edges.append((follower_id, user.id))
                        if follower_id not in queue_set and not enough_in_queue:
                            queue.append(follower_id)
                            queue_set.add(follower_id)

                for followee_id in followees:
                    if followee_id not in visited_ids:
                        edges.append((user.id, followee_id))
                        if followee_id not in queue_set and not enough_in_queue:
                            queue.append(followee_id)
                            queue_set.add(followee_id)

                iterations += 1

                if iterations % self.save_interval == 0:
                    self.state.save(self.data_path)

        except KeyboardInterrupt:
            print("\n\nInterrupt received. Terminating...")
        finally:
            self.state.save(self.data_path)

    def get_connections_list(self, cursor):
        connections_list = list()
        for page in cursor:
            connections_list.extend(page)
        return connections_list

    @staticmethod
    def get_origin_user(api, max_connections):
        print(f'Obtaining the origin user..')
        n_random_tweets = 100

        places = api.geo_search(query="USA", granularity="country")
        place_id = places[0].id

        cursor = tweepy.Cursor(
            api.search,
            q=f"place:{place_id}").items(n_random_tweets)

        tweets = limit_handler(cursor)

        users = [tweet.user for tweet in tweets]

        indices = list(range(len(users)))

        found = False
        while len(indices) > 0:
            random_index = random.randint(0, len(users) - 1)
            index = indices.pop(random_index)
            random_user = users[index]
            if (random_user.friends_count <= max_connections and random_user.followers_count <= max_connections):
                found = True
                break

        if not found:
            raise Exception('Found no users matching the requirements')

        print(f"\tId: {random_user.id}")
        print(f"\tScreen name: {random_user.screen_name}")
        print(f"\tLocation: {random_user.location}")
        print(f"\tNumber of followers: {random_user.followers_count}")
        print(f"\tNumber of followees: {random_user.friends_count}")
        print(f"\tNumber of tweets:", {random_user.statuses_count})

        return User(random_user)

