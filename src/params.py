import os

TRAIN_TEST_SPLIT = 0.8

TEST_EVERY = 1

LR = 0.001
NUM_EPOCHS = 50
BATCH_SIZE = 16

NUM_WORKERS = os.cpu_count() or 1 
