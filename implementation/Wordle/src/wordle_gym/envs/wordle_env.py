import os

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from enum import Enum
from collections import Counter
import numpy as np

import gym
from gym.spaces import Box, Discrete
#from torchrl.envs import GymLikeEnv, ActionSpace, ObservationSpace

WORD_LENGTH = 5
TOTAL_GUESSES = 6
SOLUTION_PATH = "../words/solution.csv"
VALID_WORDS_PATH = "../words/guess.csv"


class LetterState(Enum):
    ABSENT = 0
    PRESENT = 1
    CORRECT = 2


class WordleEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def _current_path(self):
        return os.path.dirname(os.path.abspath(__file__))

    def _read_solutions(self):
        return open(os.path.join(self._current_path(), SOLUTION_PATH)).read().splitlines()

    def _get_valid_words(self):
        words = []
        for word in open(os.path.join(self._current_path(), VALID_WORDS_PATH)).read().splitlines():
            words.append((word, Counter(word)))
        return words

    def __init__(self) -> None:
        self._solutions = self._read_solutions()
        self._valid_words = self._get_valid_words()
        self.action_space = spaces.Discrete(len(self._valid_words))
        self.observation_space = spaces.MultiDiscrete([3]*TOTAL_GUESSES*WORD_LENGTH)
        self.seed(0)

    def _check_guess(self, guess, guess_counter):
        c = guess_counter & self.solution_ct
        result = []
        correct = True
        reward = 0
        for idx, char in enumerate(guess):
            if c.get(char, 0) > 0:
                if self.solution[idx] == char:
                    result.append(2)
                    reward += 2
                else:
                    result.append(1)
                    correct = False
                    reward += 1
                c[char] -= 1
            else:
                result.append(0)
                correct = False
        return result, correct, reward

    def step(self, action):
        guess, guess_counter = self._valid_words[action]
        if guess in self.guesses:
            return self.obs, -1, False, {}
        self.guesses.append(guess)
        result, correct, reward = self._check_guess(guess, guess_counter)
        done = False

        for i in range(self.guess_no*WORD_LENGTH, self.guess_no*WORD_LENGTH + WORD_LENGTH):
            self.obs[i] = result[i - self.guess_no*WORD_LENGTH]

        self.guess_no += 1
        if correct:
            done = True
            reward = 1200
        if self.guess_no == TOTAL_GUESSES:
            done = True
            if not correct:
                reward = -15
        return self.obs, reward, done, {}

    def reset(self, seed=0):
        self.solution = self._solutions[np.random.randint(len(self._solutions))]
        self.solution_ct = Counter(self.solution)
        self.guess_no = 0
        self.guesses = []
        self.obs = np.zeros(TOTAL_GUESSES*WORD_LENGTH, dtype = np.int32)
        return self.obs

    def render(self, mode = "human"):
        m = {
            0: "\033[0m",
            1: "\033[30m\033[43m",
            2: "\033[30m\033[42m"
        }

        for g, o in zip(self.guesses, np.reshape(self.obs, (TOTAL_GUESSES, WORD_LENGTH))):
            o_n = ""
            for char, rwd in zip(g, o):
                o_n += m[rwd] + char + "\033[0m"
            print(o_n)

    def close(self):
        pass


# class WordleEnvWrapper(WordleEnv, EnvBase):
#     def __init__(self, **kwargs) -> None:
#         super().__init__(self, **kwargs)
        
#     def reset(self):
#         return self.env.reset()
    
#     def step(self, action):
#         return self.env.step(action)
    
#     def render(self, mode='human'):
#         return self.env.render(mode=mode)

if __name__ == "__main__":
    env = WordleEnv()
    print(env.action_space)
    print(env.observation_space)
    print(env.solution)
    print(env.observation_space.shape[0])
    print(env.step(0))
    print(env.step(0))
    print(env.step(0))
    print(env.step(0))
    print(env.step(0))
    print(env.step(0))
