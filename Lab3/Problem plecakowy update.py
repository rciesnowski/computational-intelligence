import pygad
import numpy
import time

S = [[100,7, "zegar"], [300,7, "pejzaz"], [200,6, "portret"], [40,2, "radio"], [500,5, "laptop"], [70,6, "lampka"], [100,1, "sztucce"], [250,3, "porcelana"], [300,10, "figura"], [280,3, "torebka"], [300,15, "odkurzacz"]]

#definiujemy parametry chromosomu
#geny to liczby: 0 lub 1
gene_space = [0, 1]

#definiujemy funkcje fitness
def fitness_func(solution, solution_idx):
    if (numpy.sum(solution * [el[1] for el in S]) <= 25):
        fitness = numpy.sum(solution * [el[0] for el in S])
    else:
        fitness = 0
    #lub: fitness = 1.0 / (1.0 + numpy.abs(sum1-sum2))
    return fitness

fitness_function = fitness_func

#ile chromsomĂłw w populacji
#ile genow ma chromosom
sol_per_pop = 10
num_genes = len(S)

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 5
num_generations = 30
keep_parents = 2

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 8

#inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty


#uruchomienie algorytmu
czas = time.time()
czas = czas - czas
for i in range(10):
    ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       stop_criteria=["reach_1600.0"])
    start = time.time()
    ga_instance.run()
    czas += (time.time() - start)
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(i+1)
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Number of generations passed is {generations_completed}\n".format(generations_completed=ga_instance.generations_completed))

print("Time passed (average): ", czas/10)
