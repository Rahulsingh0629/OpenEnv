from collections import deque
import random

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def store(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)