import logging


def create_logger(logname, filename, level=logging.INFO, stream=True):
    l = logging.getLogger(logname)
    formatter = logging.Formatter(
        fmt='[%(asctime)s][%(filename)10s][line:%(lineno)4d][%(levelname)4s] %(message)s',
        datefmt='%m-%d %H:%M:%S'
    )
    fh = logging.FileHandler(filename)
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if stream:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)
    return l
