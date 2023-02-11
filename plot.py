import matplotlib.pyplot as plt
import numpy as np

def calculate_probabilities(model1_outputs:list, model2_outputs:list):
    """The output lists are lists of the functions that the models learnt
    
    Returns a arrays of the probabilities of each function being learnt for all
    functions that were present in both inputs"""
    functions_in_common = set(model1_outputs).intersection(set(model2_outputs))

    model1_probabilities = []
    model2_probabilities = []
    for f in functions_in_common:
        #probability for the function is number of times it occurs divided
        #by the total number of functions learnt by the model
        model1_probabilities.append(model1_outputs.count(f)/len(model1_outputs))
        model2_probabilities.append(model2_outputs.count(f)/len(model2_outputs))

    return np.array(model1_probabilities), np.array(model2_probabilities)


#Plot probabilities for some random data
p1, p2 = calculate_probabilities([1,1,2,2,2,2,3,4,4,4], [1,1,2,2,2,2,3,4,4,4,4,4,4,4,4,4,4,4,4,4])
plt.scatter(p1,p2)
plt.xlabel("P(f|Model1)")
plt.ylabel("P(f|Model2)")
plt.title("Correlation: " + str(np.corrcoef(p1, p2)[0,1]))
plt.show()
