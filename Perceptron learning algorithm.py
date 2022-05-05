import numpy  as np
import matplotlib.pyplot as plt
def perceptron_learn(data_in):
    # Run PLA on the input data
    #
    # Inputs: data_in: Assumed to be a matrix with each row representing an
    #                (x,y) pair, with the x vector augmented with an
    #                initial 1 (i.e., x_0), and the label (y) in the last column
    # Outputs: w: A weight vector (should linearly separate the data if it is linearly separable)
    #        iterations: The number of iterations the algorithm ran for

    # Your code here, assign the proper values to w and iterations:
       w=np.zeros(len(data_in[0][0]))
       w[0]=0
       iterations=0
       while True:
            right=0
            for data in data_in:
                if (np.dot(w,data[0])>=0 and data[1]<0)or(np.dot(w,data[0])<0 and data[1]>0):
                    iterations+=1
                    w+=np.dot(data[1],data[0])
                    break
                if (np.dot(w,data[0])>=0 and data[1]>0)or(np.dot(w,data[0])<0 and data[1]<0): 
                    right+=1
            if right==np.array(data_in).shape[0]:
                iterations+=1
                break
       return w, iterations
def perceptron_experiment(N, d, num_exp):
    # Code for running the perceptron experiment in HW0
    # Implement the dataset construction and call perceptron_learn; repeat num_exp times
    #
    # Inputs: N is the number of training data points
    #         d is the dimensionality of each data point (before adding x_0)
    #         num_exp is the number of times to repeat the experiment
    # Outputs: num_iters is the # of iterations PLA takes for each experiment
    #          bounds_minus_ni is the difference between the theoretical bound and the actual number of iterations
    # (both the outputs should be num_exp long)
      
    # Initialize the return variables
    num_iters = np.zeros((num_exp,))
    bounds_minus_ni = np.zeros((num_exp,)) 
    for i in range(num_exp):
         w=np.random.rand(d+1)
         w[0]=0
         w_val=np.linalg.norm(w)
         D=[]
         rou=float('inf')
         R=0
         for j in range(N):
             x=np.random.uniform(-1,1,d+1)
             x[0]=1
             x_val=np.linalg.norm(x)
             if np.dot(x,w)>=0:
                  y=1
             else:
                  y=-1
             D.append([x,y])
             if np.dot(x,w)*y<rou:
                    rou=np.dot(x,w)*y
             if R<x_val:
                R=x_val
         num_iters[i]=perceptron_learn(D)[1]
         theoretical=R*R*w_val*w_val/(rou*rou)
         bounds_minus_ni[i]=theoretical-perceptron_learn(D)[1]
    # Your code here, assign the values to num_iters and bounds_minus_ni:
    return num_iters, bounds_minus_ni
def main():
    print("Running the experiment...")
    num_iters, bounds_minus_ni = perceptron_experiment(100, 10, 1000)

    print("Printing histogram...")
    plt.hist(num_iters)
    plt.title("Histogram of Number of Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Count")
    plt.show()

    print("Printing second histogram")
    plt.hist(np.log(bounds_minus_ni))
    plt.title("Bounds Minus Iterations")
    plt.xlabel("Log Difference of Theoretical Bounds and Actual # Iterations")
    plt.ylabel("Count")
    plt.show()

if __name__ == "__main__":
    main()
