import numpy as np
from sklearn.metrics import f1_score

task = 'informative'
algo = 'multihead-apm-5-no-low'
run = 1

if task == 'humanitarian':
    class_counts = np.array([504, 235, 126, 81, 9])

    if algo == 'fixmatch-supervised':
        if run == 1:
            confusion_matrix = np.array([[0.90239044, 0.02788845, 0.0438247,  0.02191235, 0.00398406],
            [0.17021277, 0.81276596, 0.00425532, 0.01276596, 0.        ],
            [0.112,      0.008,      0.864,      0.008,      0.008     ],
            [0.03703704, 0.,         0.04938272, 0.88888889, 0.02469136],
            [0.11111111, 0.,         0.11111111, 0.11111111, 0.66666667]])
    elif algo == 'multihead-apm-5-no-low':
        if run == 1:
            confusion_matrix = np.array([[0.94422311, 0.02191235, 0.02191235, 0.01195219, 0.        ],
            [0.14893617, 0.82978723, 0.00851064, 0.00851064, 0.00425532],
            [0.136,      0.,         0.864,      0.,         0.,        ],
            [0.03703704, 0.,         0.,         0.92592593, 0.03703704],
            [0.22222222, 0.,         0.22222222, 0.,         0.55555556]
            ])

    else:
        assert False

elif task == 'informative':
    class_counts = np.array([1030, 504])

    if algo == 'fixmatch-supervised':
        if run == 1:
            confusion_matrix = np.array([[0.95224172, 0.04775828],
                                        [0.1812749,  0.8187251 ]])

    elif algo == 'multihead-apm-5-no-low':
        if run == 1:
            confusion_matrix = np.array([[0.9454191,  0.0545809 ],
                                        [0.14940239, 0.85059761]])
    
    else:
        assert False
else:
    assert False


# Calculate F1 score for each class
true_labels = []
pred_labels = []

for i in range(len(confusion_matrix)):
    true_labels.extend([i] * class_counts[i])
    for j in range(len(confusion_matrix[i])):
        pred_labels.extend([j] * round(confusion_matrix[i][j] * class_counts[i]))

assert len(true_labels) == len(pred_labels)

# Using scikit-learn to calculate the overall F1 score (macro, micro, or weighted)
f1_macro = f1_score(true_labels, pred_labels, average='macro')
f1_micro = f1_score(true_labels, pred_labels, average='micro')
f1_weighted = f1_score(true_labels, pred_labels, average='weighted')

print(f'F1 Score (Macro): {f1_macro}')
print(f'F1 Score (Micro): {f1_micro}')
print(f'F1 Score (Weighted): {f1_weighted}')