# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
<img width="820" height="540" alt="image" src="https://github.com/user-attachments/assets/13562e03-5a60-470d-b4e9-729abe5c8448" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: VD Natchathira

### Register Number: 212224230178

```
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv(r"C:\Users\acer\Downloads\Exp-1.csv")
df
x=df[["Input"]].values
y=df[["Output"]].values
xt,xst,yt,yst=train_test_split(x,y,test_size=0.2,random_state=52) # check if requires_grad is true, false if not directly specified
scale1=MinMaxScaler()
scale2=MinMaxScaler()
xt=scale1.fit_transform(xt)
xst=scale1.transform(xst)

xt=torch.FloatTensor(xt)
xst=torch.FloatTensor(xst)
yt=torch.FloatTensor(yt)
yst=torch.FloatTensor(yst)
class neuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)


        )
    def forward(self, x):
        return self.network(x)

model=neuralNetwork()
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

epochs=5000
losses=[]

for i in range(epochs):
    optimizer.zero_grad()
    predictions=model(xt)
    loss=criterion(predictions,yt)
    loss.backward()
    optimizer.step()
    if i%100==0:
        print(f"epoch: {i}, loss: {loss.item()}")
        losses.append(loss.item())
# %% prediction
new_data=torch.FloatTensor([[16]])
new_data=torch.Tensor(scale1.transform(new_data))
predictions=model(new_data)
print(predictions.item())
# %% test
xst = torch.FloatTensor(xst)
yst = torch.FloatTensor(yst)
with torch.no_grad():
    predictions_l=model(xst)
    test_loss=criterion(predictions_l,yst)
    print(f"test loss: {test_loss.item()}")

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss during training")
plt.show()
```

### Dataset Information
<img width="203" height="587" alt="image" src="https://github.com/user-attachments/assets/fcd01eaf-de46-4af6-bd36-59a2ac79d584" />


### OUTPUT
<img width="261" height="540" alt="image" src="https://github.com/user-attachments/assets/6eda3eb2-3f6a-4cd4-8842-1fdb9d9931ef" />


### Training Loss Vs Iteration Plot
<img width="645" height="501" alt="image" src="https://github.com/user-attachments/assets/dc37d475-5fa3-438e-8bef-9e420ec3acc6" />


### New Sample Data Prediction
<img width="233" height="35" alt="image" src="https://github.com/user-attachments/assets/f7d0ae91-6362-4b1b-bb75-0bc6d4dda897" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
