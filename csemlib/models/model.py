import abc
import os

import xarray


class Model(metaclass=abc.ABCMeta):
    """
    An abstract base class handling an external Earth model.
    """

    def __init__(self):
        pass

    @abc.abstractproperty
    def data(self):
        pass

    @abc.abstractmethod
    def read(self):
        pass

    @abc.abstractmethod
    def write(self):
        pass

