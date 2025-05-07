class InvalidAudioFormatError(Exception):
    def __init__(self, detail: str):
        self.detail = detail

class InternalSenseVociceError(Exception):
    def __init__(self, detail: str):
        self.detail = detail