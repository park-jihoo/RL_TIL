from gym.envs.registration import register

register(
    id="Wordle-v0", entry_point="wordle_gym.envs.wordle_env:WordleEnv"
)