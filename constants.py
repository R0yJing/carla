

AUGMENTATION_BATCH_SIZE = 32
IM_HEIGHT = 88
IM_WIDTH = 200
NUM_EPOCHS = 5
#BATCH_SIZE = 32 #bsize / ep_len = mintimesteps per batch
TRAIN_BATCH_SIZE = 180
EP_LEN = 10
EPISODIC_BUFFER_LEN = 120 #1000/20*2
MIN_TIMESTEPS_PER_BATCH = EPISODIC_BUFFER_LEN * 5 #not sure
NUM_AGENT_TRAIN_STEPS_PER_ITER = 1000
#replay buffer contains a sequence of trajectories
N_ITER = 10
#in gigabytes
STORAGE_LIMIT = 10
MAX_TEST_DATA_SIZE = 16128
NUM_SAMPLES_PER_ITER = 21400
NUM_SAMPLES_PER_COMMAND_PER_ITER = NUM_SAMPLES_PER_ITER // 3
MAX_REPLAY_BUFFER_SIZE = NUM_SAMPLES_PER_ITER * 10 # 95200 * 4#161289

BENCHMARK_LIMIT = 0
SAMPLE_TIME=0.2
TARGET_TOLERANCE = 3
MAX_TEST_BRANCH_BUFFER_SIZE = int(MAX_REPLAY_BUFFER_SIZE / 3 * 0.33)
TOTAL_SAMPLE_TIME =  3600#47.6 * 60
MINI_BATCH_SIZE = 180
LAMBDA = 0.889
DISTANCE_BETWEEN_DEST_AND_SRC = 1000

WAYPOINT_TIMEOUT = 5
COLLISION_TIMEOUT = 3

NOISE_DURATION = 2.5

TARGET_SPEED = 30

MIN_SPEED = 3
BATCH_SIZE = 120


#71032 bytes per sample
