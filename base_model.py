class BaseModel:
    def __call__(self, frame):
        raise NotImplementedError

    def draw(self, frame):
        raise NotImplementedError

    def get_detections(self):
        raise NotImplementedError
