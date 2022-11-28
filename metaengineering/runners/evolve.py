"""
2-input XOR example -- this is most likely the simplest possible example.
"""
import os
import sys

sys.path.append("..")

from sklearn.metrics import mean_absolute_error
from src.utils.utils import get_generator, get_project_root
from src.orchestrator.config import ExplanationConfig, RunConfig
from src.pipeline.config import DataLoaderConfig, TaskLoaderConfig
from src.pipeline.taskloader import TaskLoader, TaskFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from src.utils.utils import build_config
from src.settings.tier import Tier
from src.settings.strategy import Strategy
from src.orchestrator.orchestrator import Orchestrator
from src.pipeline.dataloader import DataLoader
import visualize
import neat
from src.orchestrator.trainer import Trainer
import pickle


# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]

DataLoader.DATA_FOLDER = '/home/tijmen/tudelft/thesis/metaengineering/data/training/'


def get_fitness_function():
    tier = Tier.TIER0
    strategy = Strategy.ALL

    dl_config = DataLoaderConfig(
        additional_filters=["is_precursor", ],
        additional_transforms=["log_fold_change_protein", ]
    )

    tl_config = TaskLoaderConfig(
        data_throttle=1,
        tier=tier,
    )

    dl = DataLoader()
    dl.prepare_dataloader(dl_config)

    tl = TaskLoader()
    tl.prepare_taskloader(tl_config)

    gen = get_generator(dl, tl, strategy, tier)
    tf = next(gen)

    trainer = Trainer()
    X_train, X_test, y_train, y_test = trainer.do_train_test_split(tf, strategy)
    X_train = X_train.drop(['KO_ORF', 'metabolite_id'], axis=1)

    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            # genome.fitness = 4.0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            # print(f"{X_train.values.shape=} {y_train.values.reshape((-1, 1)).shape=}")
            for x, y in zip(X_train.values, y_train.values.reshape((-1, 1))):
                output = net.activate(x)
                # print(f"fitness: {-1 * mean_absolute_error(y, output)}")
                genome.fitness = -1 * mean_absolute_error(y, output)

    return eval_genomes


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(get_fitness_function(), 40)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)
    
    
    # for xi, xo in zip(xor_inputs, xor_outputs):
    #     output = winner_net.activate(xi)
    #     print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    # node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    # visualize.draw_net(config, winner, True, node_names=node_names)
    # visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(get_fitness_function(), 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)
