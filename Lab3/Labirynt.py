import pygad
import numpy
import time

labirynt = [["X" for i in range(12)],
            ["X", "S", "_", "_", "X", "_", "_", "_", "X", "_", "_", "X"],
            ["X", "X", "X", "_", "_", "_", "X", "_", "X", "X", "_", "X"],
            ["X", "_", "_", "_", "X", "_", "X", "_", "_", "_", "_", "X"],
            ["X", "_", "X", "_", "X", "X", "_", "_", "X", "X", "_", "X"],
            ["X", "_", "_", "X", "X", "_", "_", "_", "X", "_", "_", "X"],
            ["X", "_", "_", "_", "_", "_", "X", "_", "_", "_", "X", "X"],
            ["X", "_", "X", "_", "_", "X", "X", "_", "X", "_", "_", "X"],
            ["X", "_", "X", "X", "X", "_", "_", "_", "X", "X", "_", "X"],
            ["X", "_", "X", "_", "X", "X", "_", "X", "_", "X", "_", "X"],
            ["X", "_", "X", "_", "_", "_", "_", "_", "_", "_", "E", "X"],
            ["X" for i in range(12)]]

#definiujemy parametry chromosomu
#geny to liczby: 0 lub 1
gene_space = [0, 1, 2, 3]

#definiujemy funkcje fitness
def fitness_func(solution, solution_idx):
    x = 1
    y = 1
    licznik = 0
    for i in solution:
        licznik = licznik + 1
        if i == 0:
            x = x - 1
        elif i == 1:
            x = x + 1
        elif i == 2:
            y = y - 1
        elif i == 3:
            y = y + 1
        if (labirynt[y][x] == "X") or (licznik == 30):
            fitness = (y - 10) + (x - 10)
            break
        elif (labirynt[y][x] == "E"):
            fitness = 30 - licznik
    return fitness

fitness_function = fitness_func

#ile chromsomĂłw w populacji
#ile genow ma chromosom
sol_per_pop = 200
num_genes = 30

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 100
num_generations = 300
keep_parents = 10

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
                       stop_criteria=["reach_0"])
    start = time.time()
    ga_instance.run()
    czas += (time.time() - start)
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(i+1)
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Number of generations passed is {generations_completed}\n".format(generations_completed=ga_instance.generations_completed))

print("Time passed (average): ", czas/10)
