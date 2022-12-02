"""
2-input XOR example -- this is most likely the simplest possible example.
"""
import os
import sys
import multiprocessing
import pickle

sys.path.append("..")

from src.utils.utils import get_generator, get_project_root
from src.orchestrator.config import ExplanationConfig, RunConfig
from src.pipeline.config import DataLoaderConfig, TaskLoaderConfig
from src.pipeline.taskloader import TaskLoader, TaskFrame
from src.utils.utils import build_config
from src.settings.tier import Tier
from src.settings.strategy import Strategy
from src.orchestrator.orchestrator import Orchestrator
from src.pipeline.dataloader import DataLoader
from src.orchestrator.trainer import Trainer

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

import visualize
import neat

from neat.reporting import BaseReporter

DataLoader.DATA_FOLDER = f'{get_project_root()}/data/training/'


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
                genome.fitness -= mean_absolute_error(y, output)

    return eval_genomes

def eval_genomes(genome, config):
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
    X_test = X_test.drop(['KO_ORF', 'metabolite_id'], axis=1)

        # genome.fitness = 4.0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    # print(f"{X_train.values.shape=} {y_train.values.reshape((-1, 1)).shape=}")
    # the optimal fitness is an error of zero
    # fitness = 0 
    # for x, y in zip(X_train.values, y_train.values.reshape((-1, 1))):
    #     output = net.activate(x)
    #     # print(f"fitness: {-1 * mean_absolute_error(y, output)}")
    #     fitness -= mean_absolute_error(y, output)

    outputs = [net.activate(x)[0] for x in X_train.values]
    test_output = [net.activate(x)[0] for x in X_test.values]
    # genome.test_fitness = 
    return -1 * mean_absolute_error(y_train, outputs), -1 * mean_absolute_error(y_test, test_output)

class TestGenome(neat.DefaultGenome):
    def __init__(self, key):
        super().__init__(key)
        self.test_fitness = 0.0

class TestReporter(neat.StdOutReporter):
    def __init__(self, show_species_detail):
        super().__init__(show_species_detail)
    
    def post_evaluate(self, config, population, species, best_genome):
        super().post_evaluate(config, population, species, best_genome)
        print(f"Test performance: {best_genome.test_fitness:.2f}")
    
class TestParallelEvaluator(neat.ParallelEvaluator):
    def __init__(self, num_workers, eval_function, timeout=None):
        super().__init__(num_workers, eval_function, timeout)
    
    def evaluate(self, genomes, config):
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config)))

        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            fitness, test_fitness = job.get(timeout=self.timeout)
            genome.fitness = fitness
            genome.test_fitness = test_fitness



def run(config_file):
    # Load configuration.
    # config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
    #                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
    #                      config_file)
    config = neat.Config(TestGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(TestReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix=f"{get_project_root()}/model/{Tier.TIER0}/neat-checkpoint-"))

    # Run for up to 300 generations.
    # eval_genome = get_fitness_function()
    pe = TestParallelEvaluator(int(multiprocessing.cpu_count() / 2), eval_genomes)
    winner = p.run(pe.evaluate, 300)

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    with open(f"{get_project_root()}/model/{Tier.TIER0}/neat", 'wb') as f:
        pickle.dump(winner, f)
    
    
    # for xi, xo in zip(xor_inputs, xor_outputs):
    #     output = winner_net.activate(xi)
    #     print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    # node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    # visualize.draw_net(config, winner, view=True, filename=f"{get_project_root()}/images/explorative/neat_network")
    visualize.draw_net(config, winner, view=True, show_disabled=False)
    # visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True, filename=f"{get_project_root()}/images/performance/neat_performance.svg")
    visualize.plot_species(stats, view=True, filename=f"{get_project_root()}/images/performance/neat_species.svg")

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(get_fitness_function(), 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.ini')
    run(config_path)
