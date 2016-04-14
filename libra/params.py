from abc import ABCMeta
from abc import abstractmethod


class ShParameter(metaclass=ABCMeta):

    def __init__(self, name, value):
        self.name = name
        self.value = value

    @classmethod
    def hyphenize(cls, string):
        return '-{}'.format(string)

    def double_hyphenize(cls, string):
        return '--{}'.format(string)

    @classmethod
    def convert_params(cls, params):
        converted_param_list = []
        for param in params:
            converted_param_list.extend(param.convert())
        return converted_param_list

    @abstractmethod
    def convert(self):
        """
        Generate the string representations for the param name and value
        """


class LibraSimpleParameter(ShParameter):

    def convert(self):
        """
        If value is not None, then convert the key value pair in a pair of strings
        """
        if self.value is not None:
            return [super().hyphenize(self.name),
                    '{}'.format(self.value)]
        else:
            return []


class LibraOptionParameter(ShParameter):

    def convert(self):
        """
        An option is a parameter with a name but no value (eg. a flag like -sameev in acquery)
        For sh, only True-like options shall be specified

        """
        if self.value:
            return [super().hyphenize(self.name)]
        else:
            return []
