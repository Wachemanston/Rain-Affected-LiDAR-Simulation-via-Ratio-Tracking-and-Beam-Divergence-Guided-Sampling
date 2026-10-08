def calculate_execution_time(func):
    import time

    def warp(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print(f'[Timer] Function "{func.__name__}": {round(time.time() - start, 2)} sec')
        return res

    return warp


class classproperty(property):
    def __get__(self, obj, cls):
        return self.fget(cls)