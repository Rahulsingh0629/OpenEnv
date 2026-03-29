def flatten_obs(obs):
    return {k: v.flatten() for k, v in obs.items()}