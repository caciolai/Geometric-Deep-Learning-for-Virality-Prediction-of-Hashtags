import tweepy.models


class User:
    def __init__(self, user):
        if isinstance(user, User):
            self.id = user.id
            self.attrs = user.attrs
        elif isinstance(user, tweepy.models.User):
            self.id = user.id
            self.attrs = user
        else:
            raise TypeError(f"Passed an object of type {type(user)}.")

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, User):
            return False
        return self.id == o.id

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def __hash__(self) -> int:
        return object.__hash__(self.id)

    def __repr__(self) -> str:
        return str(self.id)

