from enum import Enum

class Innovation_type(Enum):
    ADD = 0
    ADDEXTRA = 1
    SPLIT = 2
    INIT = 3

    def __str__(self):
        return self.name