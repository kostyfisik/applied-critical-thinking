import numpy
import matplotlib.pyplot as plt
import math
from scipy.linalg import solve

#Here we choose our boundaries a and b
a = 1
b = 1

#Number of iterations
iterations=6 

Error_vector = numpy.zeros(iterations) #Create an empty vector of errors for each iteration
Step_vector = numpy.zeros(iterations) #Create an empty vector of segments for each iteration

#Our cycle for trying RMS
for q in range (1, iterations+1):
    
    #Here we decide into how many segments we divide along x and y axis
    n = 2**q #For x
    m = 2**q #For y
    
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
    
    
    Analytical_solution =numpy.zeros((n+1, m+1))
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

    
    #Calculating RMS
    RMS = (numpy.sum(numpy.power(numpy.array(Error_matrix), 2))/(n*m))**(1/2)
    Error_vector[q-1] = RMS
    
    #Finding relations between two subsequent iterations
    if q>1:
        print(Error_vector[q-2]/Error_vector[q-1])
        
    #Writing down the step
    Step_vector[q-1] = step_ratio
    
#Plotting the RMS vs step 
plt.loglog(Step_vector, Error_vector, marker='*')
plt.grid(True, which='both')