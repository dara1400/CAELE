import copy
import random
import numpy as np

#****************************************************************************** 

class Wolf:
    def __init__(self, dataset_name, fitness_function):
        model_COs = np.random.rand(5)
        model_COs /= model_COs.sum()
        self.params = model_COs
        self.dataset_name = dataset_name
        self.fitness = fitness_function(self.dataset_name, self.params)  

def gray_wolf_optimizer(dataset_name, fitness_function, max_iter, number_of_wolfs, fit_up=False):

    wolf_population = [Wolf(dataset_name, fitness_function) for i in range(number_of_wolfs)]
    wolf_population = sorted(wolf_population, key = lambda w: w.fitness, reverse=fit_up)
    alfa_wolf, beta_wolf, sigma_wolf = copy.copy(wolf_population[: 3])

    params_count = len(alfa_wolf.params)

    for iter in range(max_iter):

        if iter % 1 == 0 and iter > 1:            
            print(f'\rIter: {iter}, Best Wolf:{alfa_wolf.params}, Best Fitness: {alfa_wolf.fitness:0.6f}', end='')
            
        a = 2 * (1 - iter / max_iter) # linearly decreased from 2 to 0

        for i in range(3, number_of_wolfs):

            a1, a2, a3 = a * (2 * random.random() - 1), a * (2 * random.random() - 1), a * (2 * random.random() - 1)
            c1, c2, c3 = 2 * random.random(), 2 * random.random(), 2 * random.random()

            params1 = np.array([0.0 for i in range(params_count)])
            params2 =  np.array([0.0 for i in range(params_count)])
            params3 =  np.array([0.0 for i in range(params_count)])
            new_params =  np.array([0.0 for i in range(params_count)])

            for j in range(params_count):

                params1[j] = alfa_wolf.params[j] - a1 * abs(c1 * alfa_wolf.params[j] - wolf_population[i].params[j])
                params2[j] = beta_wolf.params[j] - a2 * abs(c2 * beta_wolf.params[j] - wolf_population[i].params[j])
                params3[j] = sigma_wolf.params[j] - a3 * abs(c3 * sigma_wolf.params[j] - wolf_population[i].params[j])
                new_params[j] += params1[j] + params2[j] + params3[j]
        
            for j in range(len(new_params)):
                new_params[j] /= 3.0
                if new_params[j] < 0.1:
                    new_params[j] = 0.1
                if new_params[j] > 0.8:
                    new_params[j] = 0.8
        
            new_params /= new_params.sum()
            new_fitness = fitness_function(dataset_name, new_params)

            if fit_up and new_fitness > wolf_population[i].fitness:
                wolf_population[i].params = new_params
                wolf_population[i].fitness = new_fitness

            if not fit_up and new_fitness < wolf_population[i].fitness:
                wolf_population[i].params = new_params
                wolf_population[i].fitness = new_fitness
        
        wolf_population = sorted(wolf_population, key = lambda w: w.fitness, reverse=fit_up)
        alfa_wolf, beta_wolf, sigma_wolf = copy.copy(wolf_population[: 3])

    return alfa_wolf
        
#******************************************************************************* 
