import gym
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import Adam
from rl.agents import DQNAgent  # pip install keras-rl2
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

print("GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

env = gym.make("SpaceInvaders-v0")


def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Conv2D(64, (8, 8), strides=(4, 4), activation='relu', input_shape=(3, height, width, channels)))
    model.add(Conv2D(32, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


height, width, channels = env.observation_space.shape
actions = env.action_space.n

model = build_model(height, width, channels, actions)


def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1, value_min=1, value_test=2,
                                  nb_steps=5000)
    memory = SequentialMemory(limit=2000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=1000)
    return dqn


dqn = build_agent(model, actions)

dqn.compile(Adam(lr=0.0001))

dqn.fit(env, nb_steps=20000, visualize=False, verbose=1)
dqn.save_weights('models/SpaceInvaders-charlie-20k.h5f', True)

# dqn.load_weights('models/SpaceInvaders-bravo-20k.h5f')
scores = dqn.test(env, nb_episodes=10, visualize=True)
print(np.mean(scores.history['episode_reward']))

#   Episodes: 10
#   name            l1  l2  steps   rate    avg     hi
#   ----------------------------------------------------
#   Agent Alpha     32  16  1500    0.0001  276     565
#   Agent Alpha     32  16  20000   0.0001  202.0   380
#   Agent Bravo     64  32  1500    0.0001  178     430
#   Agent Bravo     64  32  1500    0.001   194.5   275
#   Agent Bravo     64  32  20000   0.0001  80      165
#   Agent Charlie   128 64  1500    0.0001  365.5   510
#   Agent Charlie   128 64  20000   0.0001
#   Agent Delta     256 128 1500    0.0001  205.5   360
#   Agent Echo      512 256 1500    0.0001  72.5    205
