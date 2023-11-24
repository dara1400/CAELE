from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

#****************************************************************************** 

def calc_clf_metrics(y_pred, y_test, class_names):
    acc = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=class_names)
    cm = confusion_matrix(y_test, y_pred)
    return acc, cr, cm
  
#******************************************************************************* 
