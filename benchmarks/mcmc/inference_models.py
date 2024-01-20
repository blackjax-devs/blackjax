from inference_gym import using_jax as gym


models = {}
for target_name in gym.targets.__all__:
  if target_name in ['Banana', 'IllConditionedGaussian']:
    try:
      models[target_name] = getattr(gym.targets, target_name)()
    except:
      pass
