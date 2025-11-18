from math import fabs, pi
import random as rd


def initialize_pose():
    pose = []

    for _ in range(8):
        # Upon testing different possible angles for different joints, the following ranges make the most sense to use:
        # Coxa = -pi/4 to pi/4. Allows the leg to move left and right with respect to the body without the leg crossing into the spider's body.
        # Femur = -pi/2 to 0. Allows the leg to curl naturally without 'breaking' the joint by bending backwards.
        # Tibia = -pi/4 to 0. Same reasoning as Femur, but for the Tibia segment.
        coxa_angle = rd.uniform(-pi / 4, pi / 4)
        femur_angle = rd.uniform(-pi / 2, 0)
        tibia_angle = rd.uniform(-pi / 4, 0)

        pose.extend([coxa_angle, femur_angle, tibia_angle])

    return pose


def fitness_pose(pose):
    coxa_angles = [pose[i] for i in range(0, len(pose), 3)]
    femur_angles = [pose[i] for i in range(1, len(pose), 3)]
    tibia_angles = [pose[i] for i in range(2, len(pose), 3)]

    total_penalty = 0

    ideal_coxa_mag = pi / 12  # Aim for about 15 degrees of outward rotation

    for angle in coxa_angles:
        # Penalize deviation from the ideal magnitude
        deviation = fabs(fabs(angle) - ideal_coxa_mag)
        total_penalty += (deviation * 1.5) ** 2

    for femur in femur_angles:
        # femur near 0 (up) is bad, near -π/4 to -π/2 (down) is good
        if femur > -pi / 4:
            total_penalty += (femur + pi / 4) ** 2

    for tibia in tibia_angles:
        # tibia near 0 (up) is bad, near -π/8 to -π/4 (down) is good
        if tibia > -pi / 8:
            total_penalty += (tibia + pi / 8) ** 2

    if total_penalty < 1e-20:
        total_penalty = 1e-20

    fitness = 1.0 / (total_penalty**0.5)  # slower growth

    if fitness > 1000:
        fitness = 1000.0

    return fitness


def tournament_selection_pose(population, fitness_scores, tournament_size=2):

    pass


def crossover_pose(parent1, parent2, swap_prob=0.5):
    pass


def mutate_pose(pose, mutation_rate=0.05, mutation_strength=0.1):
    pass
