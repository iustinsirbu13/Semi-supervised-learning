
def has_argument(args, name):
    map = vars(args)
    return name in map and map[name] is not None