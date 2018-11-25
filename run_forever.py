import subprocess
import json
import rl_training

while True:
	try:
		proc_output = subprocess.getoutput('run_game.bat')
		stats = json.loads(proc_output[proc_output.find('{'):])['stats']['0']
		bot_reward = stats['score']
		bot_rank = stats['rank']
		print(bot_reward, bot_rank)
		rl_training.train()

	except KeyboardInterrupt:
		break

