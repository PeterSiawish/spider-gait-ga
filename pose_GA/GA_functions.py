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
    pass


def tournament_selection_pose(population, fitness_scores, tournament_size=2):

    pass


def crossover_pose(parent1, parent2, swap_prob=0.5):
    pass


def mutate_pose(pose, mutation_rate=0.05, mutation_strength=0.1):
    pass
