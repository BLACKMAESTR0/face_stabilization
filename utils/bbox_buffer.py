import numpy as np
from collections import deque

class BBoxBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def add(self, bbox):
        self.buffer.append(bbox)

    def ready(self):
        return len(self.buffer) == self.buffer.maxlen

    def tensor(self):
        arr = np.array(self.buffer, dtype=np.float32)
        return arr