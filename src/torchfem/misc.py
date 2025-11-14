class ClassPropertyDescriptor:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, obj, cls):
        return self.fget(cls)


def classproperty(func):
    return ClassPropertyDescriptor(func)
