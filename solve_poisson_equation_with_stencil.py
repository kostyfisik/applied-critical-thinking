import numpy
import matplotlib.pyplot as plt
import math
from scipy.linalg import solve

#Here we choose our boundaries a and b
a = 5
b = 20

#Here we decide into how many segments we divide along x and y axis
n = 50 #For x
m = 50 #For y

step_x = a / (n+1) #Length of a segment along x
step_y = b / (m+1) #Length of a segment along y
step_ratio = (step_y/step_x)**2 #This we will need later


x=numpy.linspace(0, a, n+1)
y=numpy.linspace(0, b, m+1)

#Boundary conditions
Solution_matrix=numpy.zeros(((n+1), (m+1)))

Solution_matrix[:, m] = numpy.sin(x) / numpy.sin(a) 
Solution_matrix[n, :] = numpy.sinh(y) / numpy.sinh(b) 

#Here we make A matrix
Coefficient_matrix=numpy.zeros(((m-1)*(n-1), (m-1)*(n-1)))
        
for i in range ((m-1)*(n-2)):
    Coefficient_matrix[i, i+(m-1)] = step_ratio
for i in range ((m-1)*(n-2)):
    Coefficient_matrix[i+(m-1), i] = step_ratio
    
for i in range ((n-1)*(m-1)):
    Coefficient_matrix[i, i] = -2*(1+step_ratio)
for i in range ((n-1)*(m-1)-1):
    Coefficient_matrix[i, i+1] = 1
for i in range ((n-1)*(m-1)-1):
    Coefficient_matrix[i+1, i] = 1

for i in range (n-1-1):
    Coefficient_matrix[(i+1)*(m-1)-1, (i+1)*(m-1)] = 0
    Coefficient_matrix[(i+1)*(m-1), (i+1)*(m-1)-1] = 0
    
    
#Here we make F vector 
Boundaries_vector=numpy.zeros(((n-1)*(m-1)))
Boundaries_vector[0] = - step_ratio*Solution_matrix[0, 1] - Solution_matrix[1, 0]
Boundaries_vector[m-1-1] = -step_ratio*Solution_matrix[0, m-1] - Solution_matrix[1, (m-1)+1]
Boundaries_vector[(m-1)*((n-1)-1)] = - Solution_matrix[n-1, 0] - step_ratio*Solution_matrix[(n-1)+1, 1]
Boundaries_vector[-1] = - step_ratio*Solution_matrix[(n-1)+1, m-1] - Solution_matrix[(n-1), (m-1)+1]

for i in range (m-1-2):
    Boundaries_vector[i+1] = - step_ratio*Solution_matrix[0, i+2]
    Boundaries_vector[i+1+(m-1)*(n-1-1)] = - step_ratio*Solution_matrix [(n-1)+1, i+2]
    
for i in range (n-1-2):
    Boundaries_vector[(i+1)*(m-1)] = - Solution_matrix[i+2, 0]
    Boundaries_vector[(i+1)*(m-1)+(m-1)-1] = - Solution_matrix[i+2, (m-1)+1]

#Here we find vector with our answers
Answers_vector = solve(Coefficient_matrix, Boundaries_vector)

#Here we translate vector into matrix
for i in range (n-1):
    for j in range (m-1):
        Solution_matrix[i+1, j+1] = Answers_vector[i*(m-1)+j]


Analytical_solution = numpy.zeros((n+1, m+1))
Error_matrix =numpy.zeros((n+1, m+1))

#In this cycle we will find error
for i in range(n+1):
    for j in range(m+1):
        
        #Analytical solution
        Analytical_solution[i, j] = (math.sin(x[i]) * math.sinh(y[j])) / (math.sinh(b) * math.sin(a))
        
        #Finding difference between analytical and numerical solutions
        Error_matrix[i, j] = Solution_matrix[i, j] - Analytical_solution[i, j]
        
        #The highest error
        Error_matrix = numpy.absolute(Error_matrix)
                       
fig, (ax1, ax2, ax3) = plt.subplots(figsize=(10, 5), ncols=3)
Numeric = ax1.imshow(Solution_matrix)
fig.colorbar(Numeric, ax=ax1)
ax1.set_title('Numerical')
Analytic = ax2.imshow(Analytical_solution)
fig.colorbar(Analytic, ax=ax2)
ax2.set_title('Analytical')
Error = ax3.imshow(Error_matrix)
fig.colorbar(Error, ax=ax3)
ax3.set_title('Error')
plt.show()
