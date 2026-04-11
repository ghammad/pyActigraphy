from collections import defaultdict

REGISTER = defaultdict(list)

def register(name):
    def wrapper(f):
        print('-> Function {} registered in {}'
              .format(f, name))
        REGISTER[name].append(f)
        return f
    return wrapper
