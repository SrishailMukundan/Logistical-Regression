import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("creditcard.csv", nrows=20000)

Feature_list = []
for i in range(1, 11):
    Feature_list.append(df[f'V{i}'].to_numpy())

X = np.column_stack(Feature_list)   
df['Class'] = df['Class'].astype(str).str.strip().str.replace("'", "") #Makes sure each value from the Class column is an integer
y = df['Class'].astype(int).to_numpy()[:20000] #First 20000 Rows

Features = np.column_stack(Feature_list)
def sigmoid(z): #Function for y_pred
    return 1 / (1+np.exp(-z))

losses = []
Theta = np.ones(X.shape[1]) #Theta values are initialized to 1 for all of them


def gradient_descent(X, y, Theta, lr = 0.05, iterations = 10000):
    m = len(y)
    for i in range(iterations):
        z = np.dot(X,Theta)
        y_pred = sigmoid(z)
        gradient = np.dot(X.T, (y_pred - y)) / m #Equation for gradient descent
        Theta = Theta - (lr * gradient)

        loss = -np.mean(y*np.log(y_pred) + (1-y)*np.log(1-y_pred)) #Equation for log-loss, binary cross entropy function
        losses.append(loss)
        print("Iteration:", i, "Loss:", loss)
        print("Weights:", Theta)

    return Theta

Theta_final = gradient_descent(X, y, Theta)

fig = plt.figure(figsize=(10,7))
plt.plot(range(len(losses)), losses, color='red', marker='o', alpha=0.6)
plt.ylabel("(Loss)")
plt.xlabel("(Iteration)")
plt.title("Loss vs iteration")
plt.show(block=True)
plt.close()

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(range(len(losses)), losses, Theta_final[0], color='red', marker='o', alpha=0.6)
#Change Theta_final[0] to which specific weight you want to check
ax.set_ylabel("(Loss)")
ax.set_xlabel("(Iteration)")
ax.set_zlabel("Weight 1")
ax.set_title("Weights vs Loss")
plt.show()

print("")

print("")
print("Initial Theta:", Theta)
print("Initial Loss:", 1)
print("Learning Rate:", 0.05)
print("Number of Iterations:", 10000)
print("Number of Features:", 10)
print("Final Theta:", Theta_final)
print("Final Loss (binary cross entropy):", losses[-1])

