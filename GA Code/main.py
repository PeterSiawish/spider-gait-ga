import time
import numpy as np
from GA_functions import (
    initialize_pose,
    fitness_pose,
    tournament_selection_pose,
    crossover_pose,
    mutate_pose,
)


# Chromosome: 1x24 Vector (1 pose)
def main():
    population_size = 100_200  # Clean multiple of 300
    num_generations = 100
    population = []
    fitness_scores = []

    population = [initialize_pose() for _ in range(population_size)]

    print(f"Initialized population with {population_size} chromosomes...")

    fitness_scores = [fitness_pose(member) for member in population]

    for generation in range(num_generations):
        new_population = []

        while len(new_population) < population_size:
            parent1 = tournament_selection_pose(population, fitness_scores)
            parent2 = tournament_selection_pose(population, fitness_scores)
            child1, child2 = crossover_pose(parent1, parent2)
            child1 = mutate_pose(child1)
            child2 = mutate_pose(child2)
            new_population.extend([child1, child2])

        population = new_population[:population_size]

        fitness_scores = [fitness_pose(member) for member in population]

        print(f"Generation #{generation + 1}/{num_generations} completed...")
        print(
            f"Best fitness in Generation #{generation + 1}: {max(fitness_scores):.6f}"
        )
        print(
            f"Average fitness in this Generation: {sum(fitness_scores)/len(fitness_scores)}"
        )

    print("\n===================================")
    print(f"\nEvolution completed after {num_generations} generations.")

    best_member_index = fitness_scores.index(max(fitness_scores))

    print(f"Best fitness achieved: {fitness_scores[best_member_index]:.4f}")
    print("\n===================================")

    print("Saving the evolved population of poses to 'list_of_poses.npy'...")

    np.save("list_of_poses.npy", np.array(population))

    print("Poses successfully saved.")


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time:.4f} seconds")
