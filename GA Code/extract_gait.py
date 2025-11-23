import numpy as np
from GA_functions import fitness_pose

# Once we obtain a list of 100,200 poses, we need to extract the 300 most suitable poses in order to produce a gait.

file_name = "Result Poses/list_of_poses #4 (best).npy"
list_of_poses = np.load(file_name)

print(f"List of poses shape: {len(list_of_poses)} X {len(list_of_poses[0])}")

# Select first 300
best_300_poses = list_of_poses[:300]
print(f"List of poses shape: {len(best_300_poses)} X {len(best_300_poses[0])}")

# np.save("final_gait.npy", np.array(best_300_poses))
