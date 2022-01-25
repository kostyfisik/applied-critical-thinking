import matplotlib.pyplot as plt
import numpy as np
from solve_poisson_equation_with_stencil import get_solution

max_x = 1
max_y = 1

iterations = 8
error = []
step = []
all_points = []

for q in range(1, iterations+1):
    points = 2**q
    print(f'Running simulation for {points}x{points} mesh...')
    _, _, _, error_rms, step_x, _ = \
        get_solution(max_x, max_y, total_points_x=points+2, total_points_y=points+2)
    error.append(error_rms)
    step.append(step_x)
    all_points.append(points)

slope = np.diff(np.log10(error))/np.diff(np.log10(all_points))

plt.loglog(all_points, error, marker='*')
plt.xlabel('mesh points (along x)')
plt.ylabel('error r.m.s.')
plt.title(f'a = {max_x} b = {max_y} slope @ smallest step={slope[-1]:.4f}')
plt.grid(True, which='both')
plt.savefig(f'a_{max_x}__b_{max_y}__slope__{slope[-1]:.4f}.png')
