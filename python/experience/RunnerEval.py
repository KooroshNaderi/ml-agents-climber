import os
from threading import Thread
from time import sleep


def run_learner_function(run_id, itr):
    os.system('python learn.py --run-id=' + run_id + ' --worker-id=' + str(itr) + ' --keep-checkpoints=30 --load')
    

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


def write_checkpoint_file(_model_name, _itr):
    f = open('models\\' + _model_name + '\\checkpoint', "w")
    f.write('model_checkpoint_path: "model-' + str(_itr * 50000) + '.cptk"\n')
    f.write('all_model_checkpoint_paths: "model-' + str(_itr * 50000) + '.cptk"')
    
    f.flush()
    f.close()


if __name__ == "__main__":
    os.chdir('C:\Kourosh\Project\ml-agents-0.4.0a\python')
    for p in range(1, 11):
        model_paths = [str(p) + 'e6\HumanoidUniformAround1Step', str(p) + 'e6\HumanoidUniformWall1Step',
                       str(p) + 'e6\HumanoidPathWall1Step', str(p) + 'e6\HumanoidUniformWallMultiStep']
        for m in range(1, 4):
            for i in range(1, 6):
                for itr in range(1, 21): 
                    # climber psi runner
                    write_yaml_file(i)
                    model_name = model_paths[m] + '-' + str(i)
                    write_checkpoint_file(model_name, itr)
                    if m == 1:
                        os.system('python learn.py experience\SampleUniformWall.exe --run-id=' + model_name
                                  + ' --worker-id=' + str(i) + ' --load')
                    elif m == 2:
                        os.system('python learn.py experience\SamplePathWall.exe --run-id=' + model_name
                                  + ' --worker-id=' + str(i) + ' --load')
                    else:
                        os.system('python learn.py experience\SampleUniformWallMultiStep.exe --run-id=' + model_name
                                  + ' --worker-id=' + str(i) + ' --load')
