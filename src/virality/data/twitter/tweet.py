import tweepy


class Tweet:
    def __init__(self, tweet):
        assert isinstance(tweet, tweepy.models.Status)

        self.id = tweet.id
        self.author = tweet.user.id
        self.entities = tweet.entities
        self.in_reply_to_status_id = tweet.in_reply_to_status_id
        self.full_text = tweet.full_text
        self.created_at = tweet.created_at

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Tweet):
            return False
        return self.id == o.id

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def __hash__(self) -> int:
        return object.__hash__(self.id)

    def __repr__(self) -> str:
        return f"Tweet {self.id} from {self.author}"

