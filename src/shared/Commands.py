from enum import Enum, auto

class Command(Enum):
    SEGMENT = auto()
    RETRAIN = auto()
    ANALYZE = auto()
    EXPORT = auto()
