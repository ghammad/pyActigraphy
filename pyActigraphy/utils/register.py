from collections import defaultdict

# REGISTER is intended to contain references to the exposed functions
# that can be used by external interfaces to access

REGISTER = defaultdict(list)


def register(*names):
    """
    Decorator that add functions into REGISTER
    """
    def wrapper(f):
        print('-> Function {} registered in {}'
              .format(f, names))
        for name in names:
            REGISTER[name].append(f)
        return f
    return wrapper
