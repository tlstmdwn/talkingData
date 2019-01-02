import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from multiprocessing import Pool
import time


filename = './Desktop/Kaggle_Study/all/train.csv'
num_lines = sum(1 for l in open(filename))
n=10
skip_idx = [x for x in range(1, num_lines) if x % n != 0]
talkingData_train = pd.read_csv(filename, skiprows=skip_idx)
talkingData_valid = pd.read_csv('./Desktop/Kaggle_Study/all/train_sample.csv', skiprows=skip_idx)



