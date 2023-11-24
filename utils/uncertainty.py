import torch
import numpy as np

#****************************************************************************** 

def calc_entropy(p):
    p[p<=0] = 1e-10
    pe = p * torch.log(p)
    pe = pe.sum(dim=1)
    return -pe

#****************************************************************************** 

def calc_confusion_uncertainty_matrix(y_pred, y_test, pe_entropy, pe_entropy_thresh):

    TU = len(np.where((y_pred != y_test) & (pe_entropy > pe_entropy_thresh))[0])
    FC = len(np.where((y_pred != y_test) & (pe_entropy < pe_entropy_thresh))[0])
    FU = len(np.where((y_pred == y_test) & (pe_entropy > pe_entropy_thresh))[0])
    TC = len(np.where((y_pred == y_test) & (pe_entropy < pe_entropy_thresh))[0])
  
    UAcc  = (TU + TC) / y_test.shape[0]
    
    try:
        USen = TU / (TU + FC)
    except:
        USen = None

    try:
        USpe = TC / (TC + FU)
    except:
        USpe = None
    
    try:
        UPre = TU / (TU + FU)
    except:
        UPre = None

    return TU, FC, FU, TC, UAcc, USen, USpe, UPre
    
  
#****************************************************************************** 

def calc_MCD_entropy(model, x_input, run_counts):

    model.train()
    probs = []

    for i in range(run_counts):
        y_pred_out = model(x_input)
        y_pred_softmax = torch.softmax(y_pred_out, dim=1)
        probs.append(y_pred_softmax)    

    probs = torch.stack(probs, dim=1)
    final_probs = probs.mean(dim=1)
    y_pred = torch.argmax(final_probs, dim=1)
    pe_entropy  = calc_entropy(final_probs)

    return y_pred, probs, pe_entropy, final_probs

#******************************************************************************  

def calc_ECE(y_input, y_pred, final_probs):

    all_acc_bm  = []
    all_conf_bm = []
    ECE         = 0 
    
    conf = np.max(final_probs, axis=1)                              

    for i in  np.arange(0, 1, 0.1):

        d_indexes = np.where((i <= conf) & (conf < i + 0.1))
        BM = d_indexes[0].shape[0]
        
        if BM == 0:
            AccBM  = 0
            ConfBM = 0

        else:
            acc_BM  = (y_input == y_pred)[d_indexes].sum() / BM
            conf_BM = conf[d_indexes].sum() / BM
            ECE +=  abs(acc_BM - conf_BM) * BM / y_input.shape[0]

        all_acc_bm.append(AccBM)
        all_conf_bm.append(ConfBM)

    ECE = ECE * 100

    return all_acc_bm, all_conf_bm, ECE
  
#******************************************************************************  
