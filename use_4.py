from pyexpat import model
import pandas as pd
import numpy as np
import tensorflow_hub as hub

class USE4():
    def __init__(self) -> None:
        self.model_url = "universal-sentence-encoder_4/"
        self.model = hub.load(self.model_url)
    
    def __call__(self, input):
        return self.model(input)