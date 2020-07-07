import os
import random


class ExamplePicker(object):
    def __init__(self, data_path):
        self.path = data_path
        self.examples = os.listdir(data_path)

    
    def pick_random_article(self):
        rand_index = random.randrange(0, len(self.examples))
        example = self.examples[rand_index]

        with open(os.path.join(self.path, example), 'r', encoding='utf-8') as f:
            while f.readline() != '@content\n':
                continue
            content = f.read()

            return content