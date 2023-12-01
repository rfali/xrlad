from multiprocessing import Process, JoinableQueue
import sys
from glob import glob
from os import path
from trainer_tf import main
import json
from munch import Munch

# Command-line argument for agent configurations
agent_configs = sys.argv[1]

# Create a joinable queue for task distribution
q = JoinableQueue()

# Number of threads for parallel processing
NUM_THREADS = 60

# Function to run a single configuration
def run_single_config(queue):
    while True:
        # Get a configuration path from the queue
        conf_path = queue.get()
        params = json.load(open(conf_path))
        try:
            # Call the main function with configuration parameters
            main(Munch(params))
        except Exception as e:
            print("ERROR", e)
            raise e
        queue.task_done()

# Start multiple worker processes for parallel execution
for i in range(NUM_THREADS):
    worker = Process(target=run_single_config, args=(q,))
    worker.daemon = True
    worker.start()

# Put configuration file paths into the queue for processing
for fname in glob(path.join(agent_configs, "*.json")):
    q.put(fname)

# Wait for all tasks in the queue to be completed
q.join()
