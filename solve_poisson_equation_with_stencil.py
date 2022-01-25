import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve

def get_linear_system_matrix(mesh_points_x, mesh_points_y, mesh_step_ratio):
    matrix_size = mesh_points_x * mesh_points_y
    matrix = np.zeros((matrix_size, matrix_size))
    for k in range(matrix_size):
        for i in range(matrix_size):
            if k == i:
                matrix[k, i] = -2 * (1 + mesh_step_ratio)
            if k == i + 1 and k % mesh_points_y != 0:
                matrix[k, i] = mesh_step_ratio
            if k == i - 1 and i % mesh_points_y != 0:
                matrix[k, i] = mesh_step_ratio

            if k == i + mesh_points_y  or k == i - mesh_points_y :
                matrix[k, i] = 1
    return matrix

def get_numerical_solution(max_x, max_y, total_points_x, total_points_y):
    total_shape = (total_points_x, total_points_y)

    inner_points_x = total_points_x - 2
    inner_points_y = total_points_y - 2
    all_inner_points = inner_points_x * inner_points_y
    inner_shape = (inner_points_x, inner_points_y)

    step_x = max_x / (total_points_x - 1)
    step_y = max_y / (total_points_y - 1)
    step_ratio = (step_x / step_y) ** 2

    x = np.linspace(0, max_x, total_points_x)
    y = np.linspace(0, max_y, total_points_y)
    # Boundary conditions
    total_boundary_x = np.sin(x) / np.sin(max_x)
    total_boundary_y = np.sinh(y) / np.sinh(max_y)

    inner_boundary_x = total_boundary_x[1:-1]
    inner_boundary_y = total_boundary_y[1:-1]


    linear_system_matrix = get_linear_system_matrix(inner_points_x, inner_points_y, step_ratio)
    rhs_vector = np.zeros(all_inner_points)
    for k in range(all_inner_points):
        if (k + 1) % inner_points_y == 0:
            rhs_vector[k] += -step_ratio * inner_boundary_x[int(k / inner_points_y)]
        if k > all_inner_points - inner_points_y - 1:
            rhs_vector[k] += -inner_boundary_y[k % inner_points_y]

    numerical_solution = np.zeros(total_shape)
    numerical_solution[:, -1] = total_boundary_x
    numerical_solution[-1, :] = total_boundary_y

    linear_system_solution = solve(linear_system_matrix, rhs_vector)

    numerical_solution[1:-1, 1:-1] = np.reshape(linear_system_solution, inner_shape)
    return  numerical_solution


def get_analytical_solution(max_x, max_y, total_points_x, total_points_y):
    x = np.linspace(0, max_x, total_points_x)
    y = np.linspace(0, max_y, total_points_y)
    analytical_solution = np.zeros((total_points_x, total_points_y))
    for k in range(total_points_x):
        for i in range(total_points_y):
            analytical_solution[k, i] = (np.sin(x[k]) * np.sinh(y[i])) / (np.sinh(max_y) * np.sin(max_x))
    return analytical_solution



if __name__ == '__main__':

    max_x = 5
    max_y = 4
    total_points_x = 25
    total_points_y = 15

    numerical = get_numerical_solution(max_x, max_y, total_points_x, total_points_y)
    analytical = get_analytical_solution(max_x, max_y, total_points_x, total_points_y)
    solution_error = np.abs(numerical - analytical)

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(10, 5), ncols=3)

    numerical_plot = ax1.imshow(numerical)
    fig.colorbar(numerical_plot, ax=ax1)
    ax1.set_title('Numerical')

    analytical_plot = ax2.imshow(analytical)
    fig.colorbar(analytical_plot, ax=ax2)
    ax2.set_title('Analytical')

    error_plot = ax3.imshow(solution_error)
    fig.colorbar(error_plot, ax=ax3)
    ax3.set_title('abs(error)')
    plt.show()
