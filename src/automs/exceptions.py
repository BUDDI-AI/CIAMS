class UnableToLearnBothClassesError(Exception):
    """ Exception raised when automs is unable to learn both majority and minority classes in data after several attempts """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
