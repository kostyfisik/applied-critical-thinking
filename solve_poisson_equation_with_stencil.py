import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve


def get_analytical_solution(max_x, max_y, total_points_x, total_points_y):
    x = np.linspace(0, max_x, total_points_x)
    y = np.linspace(0, max_y, total_points_y)
    analytical_solution = np.zeros((total_points_x, total_points_y))
    for k in range(total_points_x):
        for i in range(total_points_y):
            analytical_solution[k, i] = (np.sin(x[k]) * np.sinh(y[i])) / (np.sinh(max_y) * np.sin(max_x))
    return analytical_solution


def get_linear_system_matrix(mesh_points_x, mesh_points_y, mesh_step_ratio):
    matrix_size = mesh_points_x * mesh_points_y
    matrix = np.zeros((matrix_size, matrix_size))
    for k in range(matrix_size):
        for i in range(matrix_size):
            # origin for all finite differences
            if k == i:
                matrix[k, i] = -2 * (1 + mesh_step_ratio)
            # left and right finite differences
            if k == i + 1 and k % mesh_points_y != 0:
                matrix[k, i] = mesh_step_ratio
            if k == i - 1 and (k+1) % mesh_points_y != 0:
                matrix[k, i] = mesh_step_ratio
            # bottom and top finite differences
            if k == i + mesh_points_y or k == i - mesh_points_y:
                matrix[k, i] = 1

    return matrix


def get_rhs_vector(boundary_x, boundary_y, step_ratio):
    points_x = len(boundary_x)
    points_y = len(boundary_y)
    all_points = points_x * points_y
    rhs_vector = np.zeros(all_points)
    for k in range(all_points):
        if (k + 1) % points_y == 0:
            rhs_vector[k] += -step_ratio * boundary_x[int(k / points_y)]
        if k > all_points - points_y - 1:
            rhs_vector[k] += -boundary_y[k % points_y]
    return rhs_vector


def initilize_with_boundary_conditions(top_boundary, right_boundary):
    numerical_solution = np.zeros((len(top_boundary), len(right_boundary)))
    numerical_solution[:, -1] = top_boundary
    numerical_solution[-1, :] = right_boundary

    # 5-point stencil doesn't use corner points of the analytical solution boundary
    return numerical_solution, top_boundary[1:-1], right_boundary[1:-1]


def get_step(max_x, max_y, total_points_x, total_points_y):
    step_x = max_x / (total_points_x - 1)
    step_y = max_y / (total_points_y - 1)
    step_ratio = (step_x / step_y) ** 2
    return step_ratio, step_y, step_y


def get_numerical_solution(top_boundary, right_boundary, step_ratio):
    numerical_solution, boundary_x, boundary_y = \
        initilize_with_boundary_conditions(top_boundary, right_boundary)

    linear_system_matrix = \
        get_linear_system_matrix(len(boundary_x), len(boundary_y), step_ratio)

    rhs_vector = get_rhs_vector(boundary_x, boundary_y, step_ratio)

    linear_system_solution = solve(linear_system_matrix, rhs_vector)

    inner_shape = (len(boundary_x), len(boundary_y))
    numerical_solution[1:-1, 1:-1] = np.reshape(linear_system_solution, inner_shape)
    return numerical_solution


def get_solution(max_x, max_y, total_points_x, total_points_y):
    analytical = get_analytical_solution(max_x, max_y, total_points_x, total_points_y)

    top_boundary = analytical[:, -1]
    right_boundary = analytical[-1, :]
    step_ratio, step_x, step_y = get_step(max_x, max_y, total_points_x, total_points_y)

    numerical = get_numerical_solution(top_boundary, right_boundary, step_ratio)
    solution_error = np.abs(numerical - analytical)
    error_rms = (
                    np.sum(
                        np.power(
                            solution_error[1:-1, 1:-1],  # use only computed part for error estimate
                            2
                        )
                    ) / (
                        (total_points_x-2) * (total_points_y-2)
                    )
                )**0.5
    return numerical, analytical, solution_error, error_rms, step_x, step_y


if __name__ == '__main__':
    max_x = 5
    max_y = 4
    total_points_x = 25
    total_points_y = 15

    numerical, analytical, solution_error, _, _, _ = \
        get_solution(max_x, max_y, total_points_x, total_points_y)

    data = [numerical, analytical, solution_error]
    titles = ['Numerical', 'Analytical', 'abs(error)']

    fig, axs = plt.subplots(figsize=(10, 5), ncols=len(data))
    for i in range(len(data)):
        numerical_plot = axs[i].imshow(data[i])
        fig.colorbar(numerical_plot, ax=axs[i])
        axs[i].set_title(titles[i])
    plt.show()
