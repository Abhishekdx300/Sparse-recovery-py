import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime

# Problem parameters
M = 50  # Number of measurements
N = 200  # Length of the sparse vector
K = 10  # Number of non-zero elements in x

# Create the true sparse vector
np.random.seed(42)
x_true = np.zeros(N)
x_true[np.random.choice(N, K, replace=False)] = np.random.randn(K)

# Create the dictionary matrix H
H = np.random.randn(M, N)

# Create the measurement vector y with noise
noise_level = 0.01
y = H @ x_true + noise_level * np.random.randn(M)

# Genetic Algorithm setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Gene initialization
toolbox.register("attr_float", np.random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=N)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Modified fitness function with adaptive sparsity penalty
def fitness_function(individual, gen, max_gen):
    x = np.array(individual)
    residual = np.linalg.norm(y - H @ x)
    sparsity_penalty = np.sum(np.abs(x) > 1e-6)  # L0 norm approximation
    
    # Adaptive penalty: increases as generations progress
    adaptive_factor = 0.1 + 0.4 * (gen / max_gen)
    return residual + adaptive_factor * sparsity_penalty,

# Elite-guided mutation
def elite_guided_mutation(individual, elite, indpb):
    for i in range(len(individual)):
        if np.random.random() < indpb:
            if np.abs(elite[i]) > 1e-6:
                individual[i] += np.random.normal(0, 0.1)
            else:
                individual[i] = 0
    return individual,

# Local search
def local_search(individual, gen, max_gen):
    x = np.array(individual)
    for _ in range(10):  # Perform 10 iterations of local search
        i = np.random.randint(0, len(x))
        old_val = x[i]
        x[i] += np.random.normal(0, 0.1)
        new_fitness = fitness_function(x, gen, max_gen)[0]
        if new_fitness > fitness_function(individual, gen, max_gen)[0]:
            x[i] = old_val
    return list(x)

toolbox.register("evaluate", fitness_function)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Modified evolutionary algorithm
def eaModified(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(lambda ind: toolbox.evaluate(ind, 0, ngen), invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Apply elite-guided mutation to a portion of the population
        elite = tools.selBest(population, k=1)[0]
        for i in range(len(offspring) // 4):  # Apply to 25% of the population
            if np.random.random() < 0.5:
                offspring[i], = elite_guided_mutation(offspring[i], elite, indpb=0.1)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(lambda ind: toolbox.evaluate(ind, gen, ngen), invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Apply local search to the best individuals
        best_individuals = tools.selBest(offspring, k=5)
        for ind in best_individuals:
            ind[:] = local_search(ind, gen, ngen)
            ind.fitness.values = toolbox.evaluate(ind, gen, ngen)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

# Set up statistics and hall of fame
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

hof = tools.HallOfFame(1)

# Run the modified Genetic Algorithm
population = toolbox.population(n=100)
ngen = 1000
result, logbook = eaModified(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=ngen, stats=stats, halloffame=hof, verbose=True)

# Get the best solution
best_ind = hof[0]
x_recovered = np.array(best_ind)


# Create a directory for outputs
output_dir = "sparse_recovery_output1"
os.makedirs(output_dir, exist_ok=True)

# Generate a timestamp for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save results to CSV
csv_filename = os.path.join(output_dir, f"sparse_recovery_results_{timestamp}.csv")
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Index", "True_x", "Recovered_x", "Absolute_Error"])
    for i, (true, recovered) in enumerate(zip(x_true, x_recovered)):
        csv_writer.writerow([i, true, recovered, abs(true - recovered)])

print(f"Results saved to {csv_filename}")


# Plotting individual subplots

# Subplot 1: True vs Recovered Signal
plt.figure(figsize=(5, 5))
plt.stem(x_true, linefmt='b-', markerfmt='bo', label='True')
plt.stem(x_recovered, linefmt='r-', markerfmt='ro', label='Recovered')
plt.title('True vs Recovered Signal')
plt.legend()
plt.tight_layout()

# Save plot 1
true_recovered_filename = os.path.join(output_dir, f"true_recovered_{timestamp}.png")
plt.savefig(true_recovered_filename, dpi=300, bbox_inches='tight')
plt.clf()

# Subplot 2: Minimum and Average Fitness over Generations
plt.figure(figsize=(5, 5))
gen = logbook.select("gen")
fit_mins = logbook.select("min")
fit_avgs = logbook.select("avg")
plt.plot(gen, fit_mins, 'b-', label='Minimum Fitness')
plt.plot(gen, fit_avgs, 'r-', label='Average Fitness')
plt.title('Minimum and Average Fitness over Generations')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.tight_layout()

# Save plot 2
fitness_filename = os.path.join(output_dir, f"fitness_{timestamp}.png")
plt.savefig(fitness_filename, dpi=300, bbox_inches='tight')
plt.clf()

# Subplot 3: Recovery Error
plt.figure(figsize=(5, 5))

error = np.abs(x_true - x_recovered)
plt.stem(error, linefmt='g-', markerfmt='go')
plt.title('Recovery Error')
plt.xlabel('Index')
plt.ylabel('Absolute Error')
plt.tight_layout()

# Save plot 3
error_filename = os.path.join(output_dir, f"error_{timestamp}.png")
plt.savefig(error_filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"Plots saved to {true_recovered_filename}, {fitness_filename}, {error_filename}")


# Save convergence data
convergence_filename = os.path.join(output_dir, f"convergence_data_{timestamp}.csv")
with open(convergence_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Generation", "Min_Fitness", "Avg_Fitness"])
    for g, min_fit, avg_fit in zip(gen, fit_mins, fit_avgs):
        csv_writer.writerow([g, min_fit, avg_fit])

print(f"Convergence data saved to {convergence_filename}")

# Print summary statistics
print("\nSummary Statistics:")
print(f"Recovery Error: {np.linalg.norm(x_true - x_recovered)}")
print(f"Final Minimum Fitness: {fit_mins[-1]}")
print(f"Final Average Fitness: {fit_avgs[-1]}")
