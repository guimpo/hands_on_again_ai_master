import pandas as pd
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
from sklearn import preprocessing
from scipy.io.arff import loadarff
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import time
import warnings

def ClusterIndicesNumpy(clustNum, labels_array): #numpy 
        return np.where(labels_array == clustNum)[0]

# Carrega o .arff
raw_data = loadarff('./ckplus/LBP.arff')
# Transforma o .arff em um Pandas Dataframe
df = pd.DataFrame(raw_data[0])
# Imprime o Dataframe com suas colunas
df.head()

# Com o iloc voce retira as linhas e colunas que quiser do Dataframe, no caso aqui sem as classes
X = df.iloc[:, 0:-1].values

# Aqui salvamos apenas as classes agora
y = df['class']
# Substituimos os valores binários por inteiro
y_aux = []
for i in y:
    y_aux.append(int(i.decode('ascii')[-1]))
# Novo y
y = y_aux

# Dividindo o conjunto em 80% Treino e 20% Teste.
# O parâmetro random_state = 327 define que sempre será dividido da mesma forma o conjunto.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=327)

print('Tamanho do conjunto de Treino: {}'.format(X_train.shape))
print('Tamanho do conjunto de Teste: {}'.format(X_test.shape))
scaler = preprocessing.MinMaxScaler()

# Escalando os dados de treinamento
X_train = scaler.fit_transform(X_train)
# Escalando os dados de teste com os dados de treinamento, visto que os dados de teste podem ser apenas 1 amostra
X_test = scaler.transform(X_test)

# Aplicando K-Means para separar as amostras não rotuladas em clusters
def aprendizadoAtivo(X_train, X_test, y_train, y_test):
    # Inicializar o KMeans com N centroides
    num_c = 5
    kmeans = KMeans(n_clusters = int(num_c), init = 'k-means++', random_state = 1)
    print('Numero de clusters: {}'.format(int(num_c)))
    # Executar passando como parâmetro os dados
    kmeans.fit(X_train)

    # Variavel distance recebe uma tabela de distancia de cada amostra para o centroide
    distance = kmeans.fit_transform(X_train)

    # ordered_distance.append(np.argsort(distance[:,i]))

    # agrupamento das amostras em dicionário
    mydict = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

    # salvar distâncias
    cluster_distance = []
    for i in range(0, kmeans.n_clusters):
        max = mydict[i].size
        aux = []
        for j in range(0, max):
            index = mydict[i][j]
            aux.append(distance[index][i])
        cluster_distance.append(aux)

    # identificar o menor cluster e ordenar as distâncias
    sorted_index = []
    biggest_cluster_size = len(cluster_distance[0])
    for i in range(0, kmeans.n_clusters):
        if biggest_cluster_size < len(cluster_distance[i]):
            biggest_cluster_size = len(cluster_distance[i])
        sorted_index.append(np.argsort(cluster_distance[i])[::-1])
        
    # Criando as variáveis do Aprendizado Ativo fora do ciclo de iteração
    active_data = []
    active_label = []
    acc_train = []
    acc_test = []
    f1score = []
    precision = []
    recall = []
    
    print('active_data len: {}'.format(len(active_data)))
    print('active_label len: {}'.format(len(active_label)))
    # Selecionando amostras a partir do agrupamento
    for item in range(0, biggest_cluster_size):
        for cluster in range(0, kmeans.n_clusters):
            if len(sorted_index[cluster]) > item:
                index = sorted_index[cluster][item]
                active_data.append(X_train[index])
                active_label.append(y_train[index])
                print('active_data len: {}'.format(len(active_data)))
                print('active_label len: {}'.format(len(active_label)))
            
        # Transforma elas em ndarray novamente
        active_data = np.asarray(active_data)
        active_label = np.asarray(active_label)

        # Neural Net
        t = time.time()
        nnet = MLPClassifier(alpha = 1, random_state = 1)
        model8 = nnet.fit(active_data, active_label)
        print('Treino do Neural Net Terminado. (Tempo de execucao: {})'.format(time.time() - t))
        print('')

        # Neural Net

        # Variavel para armazenar o tempo
        t = time.time()
        # Usando o modelo para predição das amostras de teste
        aux = nnet.predict(X_test)
        # Método para criar a matriz de confusão
        cm = confusion_matrix(y_test, aux)
        # Método para calcular o valor F1-Score
        f1score.append(f1_score(y_test, aux, average = 'macro'))
        # Método para calcular a Precision
        precision.append(precision_score(y_test, aux, average = 'macro'))
        # Método para calcular o Recall
        recall.append(recall_score(y_test, aux, average = 'macro'))
        # Salvando as acurácias nas listas
        acc_train.append(nnet.score(active_data, active_label))
        acc_test.append(nnet.score(X_test, y_test))
        print('Acuracia obtida com o Neural Net no Conjunto de Treinamento: {:.2f}'.format(acc_train[-1]))
        print('Acuracia obtida com o Neural Net no Conjunto de Teste: {:.2f}'.format(acc_test[-1]))
        # print('Matriz de Confusão:')
        print(cm)
        print('Precision: {:.5f}'.format(precision[-1]))
        print('Recall: {:.5f}'.format(recall[-1]))
        print('F1-score: {:.5f}'.format(f1score[-1]))
        print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
        active_data = active_data.tolist()
        active_label = active_label.tolist()

    return { 'acc_train': acc_train, 
             'acc_test': acc_test,
             'f1score': f1score,
             'precision': precision,
             'recall': recall }

results = aprendizadoAtivo(X_train, X_test, y_train, y_test)
print(results['acc_test'][-1])
