import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
import pydotplus
train = pd.read_csv('training.csv')
test = pd.read_csv('test.csv')


genero1 = []


for i in train.index:
 if int(train["pop"][i]) == 1:
  # print(f'{i}--{train["pop"][i]}')
  genero1.append(1)
 elif int(train["rock"][i]) == 1:
    # print(f'{i}--{train["rock"][i]}')
  genero1.append(2)
 elif int(train["hip hop"][i]) == 1:
    # print(f'{i}--{train["hip hop"][i]}')
  genero1.append(3)
 elif int(train["Dance/Electronic"][i]):
  #  print(f'{i}')
  genero1.append(4)

train=train.assign(genero=genero1)
# print(train)
train = train.drop(columns=['pop','rock','hip hop','Dance/Electronic'])
# print(train)

genero2 = []
for i in test.index:
 if int(test["pop"][i]) == 1:
  # print(f'{i}--{test["pop"][i]}')
  genero2.append(1)
 elif int(test["rock"][i]) == 1:
    # print(f'{i}--{test["rock"][i]}')
  genero2.append(2)
 elif int(test["hip hop"][i]) == 1:
    # print(f'{i}--{test["hip hop"][i]}')
  genero2.append(3)
 elif int(test["Dance/Electronic"][i]):
  #  print(f'{i}')
  genero2.append(4)

test=test.assign(genero=genero2)
# print(test)
test = test.drop(columns=['pop','rock','hip hop','Dance/Electronic'])
# print(test)
#1:pop	2:rock	3:hip hop	4:Dance/Electronic

# train.columns = ['label',
#                    'alcohol', 
#                    'malic_acid', 
#                    'ash', 
#                    'alcalinity_of_ash', 
#                    'magnesium', 
#                    'total_phenols', 
#                    'flavanoids', 
#                    'nonflavanoid_phenols', 
#                    'proanthocyanins', 
#                    'color_intensity', 
#                    'hue',
#                    'OD280/OD315',
#                    'proline']

# print(f'{train}')
from sklearn.model_selection import train_test_split

# x = train.values[:, 1:]
# y = train.values[:, 0] # a primeira coluna do train indica a origem do vinho 

x_train=train.values[:,:-1]
y_train=train.values[:,-1]
y_train=y_train.astype('int')

x_test=test.values[:,:-1]
y_test=test.values[:,-1]
y_test=y_test.astype('int')



# print(f'{x_train}')
# print(f'{y_train}')

# print(f'{x_test}')
# print(f'{y_test}')

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(f'{x_train}')
# print(f'{x_test}')

#treinamento
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def train_model(height):
  model = DecisionTreeClassifier(criterion = 'entropy', max_depth = height, random_state = 0)
  model.fit(x_train, y_train)
  return model


#avaliacao
for height in range(1, 21): # 1-20
  model = train_model(height)
  y_pred = model.predict(x_test)
  
  print('--------------------------------------------------------------\n')
  print(f'Altura - {height}\n')
  print("Precisão: " + str(accuracy_score(y_test, y_pred)))
  
  #exportar a arvore
from IPython.display import Image 
from sklearn.tree import export_graphviz

model = train_model(3)


feature_names = ['artist' , 'duration_ms' , 'explicit' , 'year' , 'popularity' , 'danceability' , 'energy' , 'key' , 'loudness' , 'mode' , 'speechiness' , 'acousticness' , 'instrumentalness' , 'liveness' , 'valence'  ,  'tempo' ]


classes_names = ['%.f' % i for i in model.classes_]
# print(f'{classes_names}')

dot_data = export_graphviz(model, filled=True, feature_names=feature_names,class_names=classes_names, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())
graph.write_png("tree.png")
Image('tree.png')