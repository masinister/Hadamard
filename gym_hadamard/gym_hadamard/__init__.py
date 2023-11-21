from gymnasium.envs.registration import register

register(
    id='gym_hadamard/hadamard-v0',
    entry_point='gym_hadamard.envs:HadamardEnv',
)