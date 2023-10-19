from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis



def get_dataset(dataset_name):
    '''get_dataset in uci
    
    Args:
        dataset_name (int | str):  {'heart_disease', ''}
            name or id of dataset
        
    Returns:
        result (dotdict): object containing dataset metadata, dataframes, and variable info in its properties
    '''

    dataset_dic = {
        "heart_disease":45,
        "breast_cancer_wisconsin_diagnostic":17,
    }
    _id = -1 
    
    if isinstance(dataset_name,str):
        _id =  dataset_dic.get(dataset_name,-1)
    
    if _id == -1:
        raise "dataset not found"
    
    return fetch_ucirepo(id=_id)
     
def split_dataset(uci_dataset,_test_size=0.33):
    '''split_dataset
    
    Args:
        uci_dataset: dataset get by get_dataset()
        test_size: size of test dataset
        
    Returns:
        X_train, X_test, y_train, y_test 
    '''

    X = uci_dataset.data.features 
    y = uci_dataset.data.targets 
    return train_test_split( X, y, test_size=_test_size, random_state=42)    
    
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
        
def build(mod_name):
    """build a model by name

    Args:
        mod_name (str): {'LDA', 'LDF', ''}
            name of mod

    Returns:
        clf: a class which have fit(X, y) and predict(X)
    """
    
    if mod_name == "LDA" or  mod_name == "LDF":
        return LinearDiscriminantAnalysis()
    if mod_name == "QDA" or  mod_name == "QDF":
        return QuadraticDiscriminantAnalysis()
    
    raise "model name not found"