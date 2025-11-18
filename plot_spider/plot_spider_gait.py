import numpy as np
import matplotlib.pyplot as plt


def axis_angle_rotation_matrix(axis, angle):
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
    R = axis_angle_rotation_matrix(axis, angle)
    return R @ v


def forward_leg_kinematics2(base_pos, base_angle, joint_angles, segment_lengths):
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


def plot_spider_pose(ax, angles, forward_leg_kinematics2):
    n_legs = 8
    segment_lengths = [1.2, 0.7, 1.0]
    a, b = 1.5, 1.0
    left_leg_angles = np.deg2rad([45, 75, 105, 135])
    right_leg_angles = np.deg2rad([-135, -105, -75, -45])
    base_angles = np.concatenate((left_leg_angles, right_leg_angles))
    leg_labels = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]

    if len(angles) != n_legs * 3:
        raise ValueError("angles must be length 24 (8 legs × 3 joints).")

    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(a * np.cos(t), b * np.sin(t), np.zeros_like(t), "k-", linewidth=3)

    ax.scatter(a + 0.2, 0, 0, color="r", marker="^", s=60)

    for i in range(n_legs):
        idx = i * 3
        theta1, theta2, theta3 = angles[idx : idx + 3]
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
            label_pos[0], label_pos[1], label_pos[2] + 0.05, leg_labels[i], fontsize=8
        )


def animate_spider_gait(gait, delay=0.03):
    """
    Animate a full gait sequence (N×24 matrix).
    gait: np.ndarray of shape (num_frames, 24)
    delay: seconds between frames
    """
    num_frames, num_angles = gait.shape
    if num_angles != 24:
        raise ValueError("Each frame must have 24 joint angles.")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 0.5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-2, 2])
    plt.ion()

    for frame_idx in range(num_frames):
        ax.cla()
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.set_zlim([-2, 2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Spider Gait Animation - Frame {frame_idx+1}/{num_frames}")
        plot_spider_pose(ax, gait[frame_idx], forward_leg_kinematics2)
        plt.pause(delay)

    plt.ioff()
    plt.show()


if __name__ == "__main__":

    file_name = "final_gait.npy"

    gait = np.load(f"Result Poses/{file_name}")

    animate_spider_gait(gait, delay=0.4)  # Adjust delay for animation speed
