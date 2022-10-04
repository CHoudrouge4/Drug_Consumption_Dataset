import numpy as np

def sensitivity(TP, Pos):
    return TP/Pos

def specificity(TN, Neg):
    return TN/Neg

def false_pos_rate(FP, Neg):
    return FP/Neg

def false_neg_rate(FN, Pos):
    return FN/Pos

def precision(TP, FP):
    return TP/(TP + FP)

def compute_recalls_precisions(cm):
    print(cm)
    TP = cm[0, 0]
    print(cm[0, 1])
    FN = cm[0, 1]
    FP = cm[1, 0]
    TN = cm[1, 1]
    Pos = TP + FN
    Neg = FP + TN
    return [sensitivity(TP, Pos), specificity(TN, Neg), false_pos_rate(FP, Neg), false_neg_rate(FN, Pos), precision(TP, FP)]

def display_stat_latex_table(l):
    tex = '\\begin{table}[!h]\n'
    tex += '\\begin{tabular}{l | l | l| l| l | l}\n'
    tex += 'Model & Sensitivity & Specificity & False Positive Rate & False Negative Rate & Precision \\\\\\hline\n'
    models = ['DT', 'RF', 'SVM', 'KNN']
    for i in range(4):
        tex +=  models[i] + ' & ' + str(round(l[i][0], 4)) + ' & ' + str(round(l[i][1], 4)) + ' & ' + str(round(l[i][2], 4)) + ' & ' + str(round(l[i][3], 4)) + ' & ' + str(round(l[i][4], 4)) + '\\\\\n'
    tex += '\\end{tabular}\n'
    tex += '\\caption{}\n'
    tex += '\\end{table}\n'
    return tex


def cm_to_latex(cm):
    tex = '\\begin{tabular}{l | l | l| l}\n'
    tex +='                   & Predict $\oplus$ & Predict $\circleddash$ & \\\\\\hline\n'
    tex +='Actual $\oplus$ &' + str(cm[0][0]) + '&' + str(cm[0][1]) + '&' + str(cm[0][0] + cm[0][1]) + '\\\\ \n'
    tex +='Actual $\circleddash$ &' + str(cm[1][0]) + '&' + str(cm[1][1]) + '&' + str(cm[1][0] + cm[1][1])+ '\\\\\hline \n'
    tex +=                      '&' + str(cm[0][0] + cm[1][0]) + '&'+ str(cm[0][1] + cm[1][1]) +  '&' +  str(cm[0][0] + cm[0][1] +  cm[1][0] + cm[1][1]) + '\n'
    tex +='\end{tabular}\n'
    return tex

# it takes a list of cms
def display_cm_latex_code(l):
    tex = '\\begin{table}[!h]\n'
    names = ['DT', 'RF', 'SVM', 'KNN']
    for i in range(4):
        tex += '\\subfloat[' + names[i] + ']{\n'
        tex += cm_to_latex(l[i])
        tex += '}\n'
        tex += '\\hfill\n'

    tex += '\\caption{}\n'
    tex += '\\end{table}'
    return tex

#list of confusion matrices
def get_tex_file(l, i):
    tex = display_cm_latex_code(l)
    with open('table' + str(i)+ '.tex', 'w') as f:
        f.write(tex)

def get_stat_tex(l, i):
    tex = display_stat_latex_table(l)
    with open('stat_table' + str(i)+ '.tex', 'w') as f:
        f.write(tex)
