from enum import Enum, auto

class Command(Enum):
    SEGMENT = auto()
    RETRAIN = auto()
    STOP_TRAINING = auto()
    ANALYZE = auto()
    EXPORT = auto()
    LOAD_MODEL = auto()
    CALCULATE_REAL_IMAGE_WIDTH = auto()
    TEST_MODEL = auto()
    GET_DM_IMAGE = auto()