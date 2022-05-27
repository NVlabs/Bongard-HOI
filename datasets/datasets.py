import os


datasets = {}
def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(name, **kwargs):
    dataset = datasets[name](**kwargs)
    return dataset
