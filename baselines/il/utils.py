import re

# From: https://github.com/felipecode/coiltraine/blob/29060ab5fd2ea5531686e72c621aaaca3b23f4fb/coilutils/attribute_dict.py#L9
class AttributeDict(dict):

    IMMUTABLE = '__immutable__'

    def __init__(self, *args, **kwargs):
        super(AttributeDict, self).__init__(*args, **kwargs)
        self.__dict__[AttributeDict.IMMUTABLE] = False

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if not self.__dict__[AttributeDict.IMMUTABLE]:
            if name in self.__dict__:
                self.__dict__[name] = value
            else:
                self[name] = value
        else:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but AttributeDict is immutable'.
                format(name, value)
            )

    def immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested AttributeDict.
        """
        self.__dict__[AttributeDict.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, AttributeDict):
                v.immutable(is_immutable)
        for v in self.values():
            if isinstance(v, AttributeDict):
                v.immutable(is_immutable)

    def is_immutable(self):
        return self.__dict__[AttributeDict.IMMUTABLE]

    def __repr__(self):
        return str(self.__dict__)

def tryint(s):
    try:
        return int(s)
    except Exception:
        return s


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(lst):
    """ Sort the given list in the way that humans expect.
    """
    lst.sort(key=alphanum_key)

