import torch

def compute_MAPE(y_true, y_pred): 
    MAPE = torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100
    return MAPE.item()

def compute_MAE(y_true, y_pred):
    MAE = torch.sum(torch.abs(y_pred - y_true)) / y_true.shape[0] / y_true.shape[1]
    return MAE.item()

def choose_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")