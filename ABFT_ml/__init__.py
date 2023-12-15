from gym.envs.registration import register

register(
    id = 'MyEnv-v1',
    entry_point = 'ABFT_ml.env.environment:MyEnv',
)