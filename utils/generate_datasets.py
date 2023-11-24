import pickle
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import StratifiedShuffleSplit

#****************************************************************************** 

def generate_2moons(n_samples, noise, dataset_path):  
    x, y = make_moons(n_samples=n_samples, noise=noise)
    dataset_spliter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0) 
    train_index, test_index = next(dataset_spliter.split(x, y))     
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    with open(dataset_path,'wb') as f:
        pickle.dump([x_train, y_train, x_test, y_test], f)
        
#****************************************************************************** 

def generate_blobs(n_samples, centers, cluster_std, dataset_path):  
    x, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=0)
    dataset_spliter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0) 
    train_index, test_index = next(dataset_spliter.split(x, y))     
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    with open(dataset_path,'wb') as f:
        pickle.dump([x_train, y_train, x_test, y_test], f)
        
#****************************************************************************** 