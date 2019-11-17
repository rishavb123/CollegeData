import dill as pickle
import time
import preprocessing as pp
import matplotlib.pyplot as plt
from nn import NeuralNetwork
import numpy as np
import tensorflow as tf
import time
import multiprocessing

from util import *
from wrappers import *

labels = None
dataset = None

# Config
save = True
use_tf = False
tag = ""
should_print = True

# num_of_generations = 10
# population_size = 20
# breed_percent = 0.25

num_of_generations = 2
population_size = 1
breed_percent = 1
assert breed_percent * population_size > 0

dna_ranges = [(5, 20), (1, 4), (2, 10), (0, len(activations) - 1)]

names = ['Cornell', 'GeorgiaTech', 'Illinois', 'UMD', 'UMich']
names = ['Cornell']#, 'GeorgiaTech']
# college_name = 'Illinois'

orig_time = time.time()
datasets = {}



def print_timer(old):
    elapsed = time.time() - old
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    print("Took", h, "hours,", m, "minutes, and", s, "seconds")

def run(college_name, should_print=False):
    college_time = time.time()

    print("Starting for", college_name)

    load_data(college_name, datasets)

    results, in_func, out_func = pp.preprocess(datasets[college_name])
    input_dim = len(results[0][0])
    output_dim = len(results[0][1])

    train_data, cross_validation_data, test_data = split_data(results)

    nn = None
    dna = None
    best = None

    class DNAModel:
        
        def __init__(self, dna):
            self.dna = dna
            self.nn = None
            self.fitness_score = None
            self.accuracy = 0
            self.cost = 1000000

        def mutate(self, mutation_chance=0.1):
            for i in range(len(self.dna)):
                if np.random.random() < mutation_chance:
                    self.dna[i] = clamp(self.dna[i] + ((np.random.random() < 0.5) * 2 - 1) * int(np.random.random() * 3 + 1), dna_ranges[i])
            return self

        def compile(self):
            if use_tf:
                self.nn = make_tf_nn(self.dna, input_dim, output_dim)
            else:
                self.nn = make_nn_with_dna(self.dna, input_dim, output_dim)

        def train(self):
            if use_tf:
                self.nn.fit(np.array([pp.to_row(r[0]) for r in train_data], "float32"), np.array([pp.to_row(r[1]) for r in train_data], "float32"), epochs=self.dna[0] * 10)
            else:
                self.nn.train_set([pp.to_column(r[0]) for r in train_data], [pp.to_column(r[1]) for r in train_data], epoch=self.dna[0] * 10, with_cost=False, print_progress=False)

        def fitness(self, data=None):
            if data == None:
                if self.fitness_score != None:
                    return self.fitness_score
                data = cross_validation_data
            if use_tf:
                self.cost, self.accuracy = self.nn.evaluate(np.array([pp.to_row(r[0]) for r in data], "float32"), np.array([pp.to_row(r[1]) for r in data], "float32"))
                self.fitness_score = 1 / self.cost
            else:
                self.cost, self.accuracy = self.nn.evaluate_classification([pp.to_column(r[0]) for r in data], [pp.to_column(r[1]) for r in data])
                self.fitness_score =  1 / self.cost

            return self.fitness_score

        def copy(self):
            return DNAModel(self.dna)

        @staticmethod
        def crossover(m1, m2):
            return DNAModel([a if np.random.random() < 0.5 else b for a, b in zip(m1.dna, m2.dna)])

        @staticmethod
        def random_config():
            return DNAModel([np.random.choice(range(r[0], r[1] + 1)) for r in dna_ranges])

    population = [DNAModel.random_config() for _ in range(population_size)]

    for gen in range(num_of_generations):
        old_time = time.time()
        print(college_name, "Generation", str(gen+1) + ":")
        for i in range(population_size):
            model = population[i]
            if should_print: print("\tTraining", str(i + 1) + "/" + str(population_size))
            if should_print: print("\t\tDNA: ", dict_dna(model.dna))
            model.compile()
            model.train()
            model.fitness()
            if should_print: print("\t\tCost: ", model.cost)
            if should_print: print("\t\tAccuracy: ", model.accuracy)
        population.sort(key=lambda x:-x.fitness())
        parents = population[:int(population_size * breed_percent)]
        best = parents[0]
        total_fitness = sum([parent.fitness() for parent in parents])
        p = [parent.fitness() / total_fitness for parent in parents]
        def choose():
            return np.random.choice(parents, p=p)
        population = [DNAModel.crossover(choose(), choose()).mutate() if i < population_size * 0.8 else DNAModel.random_config() for i in range(population_size - 1)]
        population.append(best.copy())
        if should_print: print("\tBest Model:")
        if should_print: print("\t\tDNA: ", dict_dna(best.dna))
        if should_print: print("\t\tCost: ", best.cost)
        if should_print: print("\t\tAccuracy: ", best.accuracy)
        print(college_name, "Generation", gen+1, end=" ")
        print_timer(old_time)
        if should_print: print('------------------------------------------------------------------------------')

    nn = best.nn
    dna = best.dna
    best.fitness(test_data)

    print("Best Overall for " + college_name + ":")
    print("\tDNA: ", dict_dna(dna))
    print("\tCost: ", best.cost)
    print("\tAccuracy: ", best.accuracy)

    model = Model(nn, in_func, out_func) if not use_tf else TFModel(nn, in_func, out_func)
    if save:
        tag = str(int(best.accuracy * 100))
        with open('./optimized_models/model' + ('-tf-' if use_tf else '-') + college_name + "-" + ((tag + "-") if tag != None and tag != "" else "") + str(dna)[1:-1] + "-" + str(int(time.time())) + '.pkl', 'wb') as f:
            pickle.dump(model, f)
    print(college_name, end=" ")
    print_timer(college_time)
    print('-----------------------------------------------------------------------------------------------------')


if __name__ == '__main__':
    processes = [multiprocessing.Process(target=run, args=(college_name, should_print)) for college_name in names]
    running = []

    can_run = 3

    for i in range(can_run):
        if len(processes) > 0:
            p = processes.pop()
            p.start()
            running.append(p)
        else:
            break

    while True:
        if len(processes) == 0 and len(running) == 0:
            break
        for p in running:
            if not p.is_alive():
                running.remove(p)
        while len(running) < can_run and len(processes) > 0:
            p = processes.pop()
            p.start()
            p.join()
            running.append(p)
        

    print()
    print_timer(orig_time)
    print()