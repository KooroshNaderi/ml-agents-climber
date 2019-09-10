import os
from threading import Thread
from time import sleep


def run_learner_function(run_id, itr):
    if itr == 1:
        os.system('python learn.py --run-id=' + run_id + ' --train')
    else:
        os.system('python learn.py --run-id=' + run_id + ' --load --train')


def write_yaml_file(num_run):
    f = open("trainer_config.yaml", "w")
    f.write("default:\n")
    f.write("    trainer: ppo \n")
    f.write("    batch_size: 1024 \n")
    f.write("    beta: 5.0e-3 \n")
    f.write("    buffer_size: 10240 \n")
    f.write("    epsilon: 0.2 \n")
    f.write("    gamma: 0.99 \n")
    f.write("    hidden_units: 128 \n")
    f.write("    lambd: 0.95 \n")
    f.write("    learning_rate: 3.0e-4 \n")
    f.write("    max_steps: 5.0e4 \n")
    f.write("    memory_size: 256 \n")
    f.write("    normalize: false \n")
    f.write("    num_epoch: 3 \n")
    f.write("    num_layers: 2 \n")
    f.write("    time_horizon: 64 \n")
    f.write("    sequence_length: 64 \n")
    f.write("    summary_freq: 1000 \n")
    f.write("    use_recurrent: false \n")
    f.write("    use_curiosity: false \n")
    f.write("    curiosity_strength: 0.01 \n")
    f.write("    curiosity_enc_size: 128 \n")
    f.write("\n")

    f.write("ClimberBrain: \n")
    f.write("    normalize: true \n")
    f.write("    num_epoch: 3 \n")
    f.write("    time_horizon: 1000 \n")
    f.write("    batch_size: 2048 \n")
    f.write("    buffer_size: 20480 \n")
    f.write("    beta: 5.0e-3 \n")
    f.write("    learning_rate: " + str(num_run) + ".0e-4 \n")
    f.write("    gamma: 0.995 \n")
    f.write("    max_steps: " + str(num_run) + "e6 \n")
    f.write("    summary_freq: 3000 \n")
    f.write("    num_layers: 3 \n")
    f.write("    hidden_units: 256 \n")

    f.flush()
    f.close()


if __name__ == "__main__":
    os.chdir('C:\Kourosh\Project\ml-agents-0.4.0a\python')

    for i in range(3, 11):
        # climber psi runner
        write_yaml_file(i)
        thread = Thread(target=run_learner_function, args=('MultiStep', i))
        thread.start()
        sleep(5)
        os.system('experience\HumanoidClimber_MultiStep.exe')