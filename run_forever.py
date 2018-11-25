import subprocess
import json
import rl_training
from time import time
import numpy as np

trainer = rl_training.TrainAgent()
count = 0
rewards = []
while True:
	count += 1
	try:
		now = time()
		proc_output = subprocess.getoutput('run_game.bat')
		stats = json.loads(proc_output[proc_output.find('{'):])['stats']['0']
		bot_reward = stats['score']
		bot_rank = stats['rank']
		rewards.append(bot_reward)
		trainer.train()
		print("Iteration took {:.0f} seconds".format(time() - now))
		if count % 20 == 0:
			print("Mean reward over past 20 games: {}".format(np.mean(rewards)))
			rewards = []

	except KeyboardInterrupt:
		break

