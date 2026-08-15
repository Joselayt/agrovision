import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import datetime



vector = TfidfVectorizer()
encoder = LabelEncoder()



df_train = pd.read_csv('./datasets/dataset_train.csv')
df_test = pd.read_csv('./datasets/dataset_test.csv')
df = np.vstack((df_train, df_test))
df = pd.DataFrame(df, columns=[c for c, d in df_train.items()])

nombre = df['nombre']
Y = df['feature']
df = df.drop(columns=['nombre', 'feature'])

nombre = vector.fit_transform(nombre).toarray()
nombre = pd.DataFrame(nombre, columns=[f'n{i}' for i in range(nombre.shape[1])])
df['n0'] = nombre['n0']
df['n1'] = nombre['n1']

df.dtype = np.float32
x_train, x_test, y_train, y_test = train_test_split(df, Y, test_size=0.2, random_state=42)


x_train_tensor = torch.tensor(np.array(x_train.values, dtype=np.float16), dtype=torch.float32)
y_train_tensor = torch.tensor(np.array(y_train.values, dtype=np.float16), dtype=torch.long)
x_test_tensor = torch.tensor(np.array(x_test.values, dtype=np.float16), dtype=torch.float32)
y_test_tensor = torch.tensor(np.array(y_test.values, dtype=np.float16), dtype=torch.long)

class classificador(nn.Module):
    def __init__(self, input_size, features):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 5)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(5, 3)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(3, features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x


model = classificador(
    input_size=7,
    features=2
)


optimizer = optim.Adam(model.parameters(), lr=0.00001)
criterion = nn.CrossEntropyLoss()
epochs = 600000
antes = datetime.datetime.now()


err, aprendido, epoc = [], [], []

def evaluar():

    model.eval()

    with torch.no_grad():
        salida = model(x_test_tensor)
        ypred = torch.argmax(salida, dim=1)

        accuracy = accuracy_score(y_test_tensor.numpy(), ypred.numpy())

        return accuracy


def training():

    model.train()

    for epoch in range(epochs):
        salida = model(x_train_tensor)
        error = criterion(salida, y_train_tensor)

        optimizer.zero_grad()
        error.backward()
        optimizer.step()

        if epoch%10 == 0:
            apr = evaluar()
            model.train()

            err.append(error.item())
            aprendido.append(apr)
            epoc.append(epoch)

            x1.set_data(epoc, err)
            x2.set_data(epoc, aprendido)


            ax.relim()
            xx.relim()
            ax.autoscale_view()
            xx.autoscale_view()

            plt.draw()
            plt.pause(0.0001)
        dur = datetime.datetime.now() - antes
        texto.set_text(f'duration: {dur}; error: {err[-1]:.2f}; learning: {aprendido[-1]:.2f}')

plt.ion()
fig, ax = plt.subplots()
x1, = ax.plot([], [], color="red")

xx = ax.twinx()
x2, = xx.plot([],[], color="blue")
ax.set_xlabel('EPOCHS', color="green", fontsize=30)
ax.set_ylabel('CRITERION', color="red", fontsize=30)
xx.set_ylabel('LEARNING', color="blue", fontsize=30)
plt.title('TRAINING "CONTROL DE CALIDAD - MODEL"', color="green", fontsize=50)

texto = xx.text(
    0.5,
    0.9,
    '',
    transform=xx.transAxes,
    fontfamily = "monospace",
    fontsize=25,
    color="cyan",
    backgroundcolor=(0.5,0.5,0.5, 1)
)


import joblib

training()

plt.ioff()
plt.show()

torch.save(model.state_dict(), "./models/control-calidad.pt")
joblib.dump(vector, "./models/vector.gz")