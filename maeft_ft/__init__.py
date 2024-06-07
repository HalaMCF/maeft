from gym.envs.registration import register

register(
    id = 'MyEnv-v2',
    entry_point = 'maeft_ft.env.environment:MyEnv',
)