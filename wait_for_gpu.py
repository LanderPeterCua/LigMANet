#######################################
# Template script for finding an available GPU, and setting a time limit for GPU-time execution
# Script by: Neil Del Gallego
# DISCLAIMER: Script was only tested for Python commandline/terminal. Using this on notebook may not guarantee the same behavior.
# The script is also still in its early stages.
#######################################

import multiprocessing
import time
import GPUtil
from main import main as run_solver
import tensorflow as tf

def train_proper(gpu_device):
    print("Your training routine here. It will be executed on " +gpu_device)
    run_solver()

def main():
    #modify your script execution time here
    EXECUTION_TIME_IN_HOURS = 72
    execution_seconds = 3600 * EXECUTION_TIME_IN_HOURS

    GPUtil.showUtilization()
    device_id = GPUtil.getFirstAvailable(maxMemory=0.01, maxLoad=0.01, attempts=2900, interval=30, verbose=True)
    gpu_device = "cuda:" + str(device_id[0])
    print("Available GPU device found: ", gpu_device)

    p = multiprocessing.Process(target=train_proper, name="train_proper", args=(gpu_device,))
    p.start()

    time.sleep(execution_seconds)  # causes p to execute code for X seconds. 3600 = 1 hour

    # terminate
    print("--------------------------------------------------")
    print("--------------------------------------------------")
    print("\n Process " + p.name + " has finished execution.")
    print("--------------------------------------------------")
    print("--------------------------------------------------")
    p.terminate()
    p.join()

if __name__ == "__main__":
    main()