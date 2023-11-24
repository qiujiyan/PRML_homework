from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from discriminantanalysis import DiscriminentAnalysis

from KDEClassifier import KDEClassifier
from MQDFClassifier import MQDFClassifier


def get_dataset(dataset_name):
    '''get_dataset in uci
    
    Args:
        dataset_name (int | str):  {'heart_disease', ''}
            name or id of dataset
        
    Returns:
        result (dotdict): object containing dataset metadata, dataframes, and variable info in its properties
    '''

    dataset_dic = {
        "breast_cancer_wisconsin_diagnostic":17,
        "iris":53,
        "wine":109,
        'breast_cancer_wisconsin_original' :15,
        'automobile' :10,
        'ionosphere' :46,
    }
    _id = -1 
    
    if isinstance(dataset_name,str):
        _id =  dataset_dic.get(dataset_name,-1)
    
    if _id == -1:
        raise "dataset not found"
    
    return fetch_ucirepo(id=_id)
     
def split_dataset(uci_dataset,_test_size=0.33,random_state=42):
    '''split_dataset
    
    Args:
        uci_dataset: dataset get by get_dataset()
        test_size: size of test dataset
        
    Returns:
        X_train, X_test, y_train, y_test 
    '''

    X = uci_dataset.data.features 
    y = uci_dataset.data.targets 
    return train_test_split( X, y, test_size=_test_size, random_state=random_state)    
    
def format_dataset(uci_dataset):
    """format dataset, include fill nan, to numpy, reshape

    Args:
        uci_dataset (_type_): _description_

    Returns:
        _type_: _description_
    """
    uci_dataset.data.features  = uci_dataset.data.features.fillna(0).to_numpy()
    uci_dataset.data.targets  = uci_dataset.data.targets.fillna(0).to_numpy().flatten()
    
    return uci_dataset
        
def build(mod_name,**kwarg):
    """build a model by name

    Args:
        mod_name (str): {'LDA', 'LDF', ''}
            name of mod

    Returns:
        clf: a class which have fit(X, y) and predict(X)
    """
    
    if mod_name in ["LDA", "LDF"]:
        return LinearDiscriminantAnalysis()
    if mod_name in ["QDA", "QDF"]:
        return QuadraticDiscriminantAnalysis()
    if "RDA" in mod_name   or "RDF" in mod_name :
        alpha = float(mod_name.split("_")[1])
        return DiscriminentAnalysis(alpha = alpha)
    if "KDE" in mod_name   or "KW" in mod_name:
        b = float(mod_name.split("_")[1])
        return KDEClassifier(bandwidth=b)
    if "MQD" in mod_name:
        delta = float(mod_name.split("_")[1])
        return MQDFClassifier(delta=delta)
    raise "model name not found"