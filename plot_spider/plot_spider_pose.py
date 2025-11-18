import numpy as np
from math import pi
import matplotlib.pyplot as plt
import os, sys


def plot_spider_pose(angles, forward_leg_kinematics2):
    """
    Plot a static 3D spider pose based on joint angles.

    Parameters:
        angles: list or np.ndarray of 24 joint angles in radians
                [theta1_1, theta2_1, theta3_1, ..., theta1_8, theta2_8, theta3_8]
        forward_leg_kinematics2: callable function that returns (j1, j2, j3, j4)
                                 given base_pos, base_angle, [theta1, theta2, theta3], segment_lengths
    """

    n_legs = 8
    segment_lengths = [1.2, 0.7, 1.0]
    a, b = 1.5, 1.0

    left_leg_angles = np.deg2rad([45, 75, 105, 135])
    right_leg_angles = np.deg2rad([-135, -105, -75, -45])
    base_angles = np.concatenate((left_leg_angles, right_leg_angles))

    leg_labels = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]

    if len(angles) != n_legs * 3:
        raise ValueError(
            "Input angles must be a 1x24 vector (3 angles per leg for 8 legs)."
        )

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 0.5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-2, 2])
    ax.view_init(elev=30, azim=45)
    ax.grid(True)

    t = np.linspace(0, 2 * np.pi, 100)
    body_x = a * np.cos(t)
    body_y = b * np.sin(t)
    ax.plot(body_x, body_y, np.zeros_like(t), "k-", linewidth=3)

    ax.scatter(a + 0.2, 0, 0, color="r", marker="^", s=60)

    print("--- Spider Pose ---")

    for i in range(n_legs):
        idx = i * 3
        theta1, theta2, theta3 = angles[idx : idx + 3]
        print(f"Leg {leg_labels[i]}: θ1={theta1:.3f}, θ2={theta2:.3f}, θ3={theta3:.3f}")

        angle = base_angles[i]
        x_base = a * np.cos(angle)
        y_base = b * np.sin(angle)
        base_pos = np.array([x_base, y_base, 0])

        j1, j2, j3, j4 = forward_leg_kinematics2(
            base_pos, angle, [theta1, theta2, theta3], segment_lengths
        )

        ax.plot([j1[0], j2[0]], [j1[1], j2[1]], [j1[2], j2[2]], "k-", linewidth=2)
        ax.plot([j2[0], j3[0]], [j2[1], j3[1]], [j2[2], j3[2]], "b-", linewidth=2)
        ax.plot([j3[0], j4[0]], [j3[1], j4[1]], [j3[2], j4[2]], "r-", linewidth=2)
        ax.scatter(j4[0], j4[1], j4[2], color="r", s=30)

        offset = 0.2
        label_pos = base_pos + offset * np.array([np.cos(angle), np.sin(angle), 0])
        ax.text(
            label_pos[0],
            label_pos[1],
            label_pos[2] + 0.05,
            leg_labels[i],
            fontsize=10,
            fontweight="bold",
        )

    plt.show()


import numpy as np


def axis_angle_rotation_matrix(axis, angle):
    """Return 3×3 rotation matrix for rotation about 'axis' by 'angle' radians."""
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ]
    )


def rotate_vector(v, axis, angle):
    """Rotate vector v around 'axis' by 'angle' radians."""
    R = axis_angle_rotation_matrix(axis, angle)
    return R @ v


def forward_leg_kinematics2(base_pos, base_angle, joint_angles, segment_lengths):
    """
    Forward kinematics for one spider leg.

    Parameters
    ----------
    base_pos : array-like, shape (3,)
        [x, y, z] position of leg base on body.
    base_angle : float
        Angle around body ellipse where leg base is located (radians).
    joint_angles : array-like of 3 floats
        [theta1, theta2, theta3] = [coxa yaw, femur pitch, tibia pitch].
    segment_lengths : array-like of 3 floats
        [L1 (coxa), L2 (femur), L3 (tibia)].
    Returns
    -------
    (j1, j2, j3, j4) : tuple of np.ndarray
        3D coordinates of each joint.
    """

    theta1, theta2, theta3 = joint_angles
    L1, L2, L3 = segment_lengths

    j1 = np.array(base_pos, dtype=float)

    coxa_elevation = np.deg2rad(30)

    coxa_horiz_dir = np.array(
        [np.cos(base_angle + theta1), np.sin(base_angle + theta1), 0.0]
    )

    rot_axis = np.cross(coxa_horiz_dir, [0, 0, 1])
    if np.linalg.norm(rot_axis) == 0:
        rot_axis = np.array([1, 0, 0])
    R = axis_angle_rotation_matrix(rot_axis, coxa_elevation)

    coxa_dir = R @ coxa_horiz_dir

    j2 = j1 + L1 * coxa_dir

    femur_rot_axis = np.cross(coxa_dir, [0, 0, 1])
    femur_rot_axis /= np.linalg.norm(femur_rot_axis)
    femur_dir = rotate_vector(coxa_dir, femur_rot_axis, theta2)
    j3 = j2 + L2 * femur_dir

    tibia_rot_axis = np.cross(femur_dir, [0, 0, 1])
    tibia_rot_axis /= np.linalg.norm(tibia_rot_axis)
    tibia_dir = rotate_vector(femur_dir, tibia_rot_axis, theta3)
    j4 = j3 + L3 * tibia_dir

    return j1, j2, j3, j4


if __name__ == "__main__":
    base_pos = [1.5, 0, 0]
    base_angle = np.deg2rad(45)
    joint_angles = [0.0, np.deg2rad(20), np.deg2rad(-10)]
    segment_lengths = [1.2, 0.7, 1.0]

    j1, j2, j3, j4 = forward_leg_kinematics2(
        base_pos, base_angle, joint_angles, segment_lengths
    )
    print("j1:", j1)
    print("j2:", j2)
    print("j3:", j3)
    print("j4:", j4)

if __name__ == "__main__":
    test_angles = np.zeros(24)
    test_angles_zeros = [0 for _ in range(24)]
    test_angles = [
        -pi / 2,
        -pi / 2,
        0,
        0,
        -pi / 2,
        0,
        0,
        -pi / 2,
        0,
        0,
        -pi / 2,
        0,
        0,
        -pi / 2,
        0,
        0,
        -pi / 2,
        0,
        0,
        -pi / 2,
        0,
        -pi / 2,
        -pi / 2,
        0,
    ]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    npy_files = [f for f in os.listdir(script_dir) if f.endswith(".npy")]

    if not npy_files:
        sys.exit("❌ No .npy file found in the current directory.")

    pose_file = npy_files[0]

    pose_path = os.path.join(script_dir, pose_file)

    data = np.load(pose_path)

    for i in range(100):
        plot_spider_pose(data[0], forward_leg_kinematics2)
