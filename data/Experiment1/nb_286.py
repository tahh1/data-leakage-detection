#!/usr/bin/env python
# coding: utf-8

# # RAPIDS SVM for MoA
# 
# * **Version 1:** SVC (GPU)
# * **Version 2:** SVR (GPU) + LR (CPU)
# * **Version 3:** SVC (GPU) + decision function + LR (CPU)
# * **Version 4:** one seed of Version 3
# * **Version 5:** SVR (GPU) + LR (CPU) + Tuned Hyperparameter 'C's
# * **Version 6:** SVC (GPU) + decision function + LR (CPU) + Tuned Hyperparameter 'C's
# 
# RAPIDS cuML is a great library alows training sklearn models on GPU. Available classification models include Logistic Regresssion, SVC, Random Forest and KNN, etc..
# 
# Konrad has tried to train SVR models in [SVR Models][1]. In this notebook, I try training 3000+ SVC models in 2 hours on GPU, which should take forever on CPU...
# 
# [1]: https://www.kaggle.com/konradb/build-model-svm

# In[1]:


import warnings, sys
warnings.filterwarnings("ignore")

# Thanks to Chris's RAPIDS dataset, it only takes around 1 min to install offline
get_ipython().system('cp ../input/rapids/rapids.0.15.0 /opt/conda/envs/rapids.tar.gz')
get_ipython().system('cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null')
sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 
get_ipython().system('cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# In[2]:


import os
import gc
import pickle
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from tqdm.notebook import tqdm
from time import time


# # Data Preparation

# In[3]:


train_features = pd.read_csv('../input/lish-moa/train_features.csv')
train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
test_features = pd.read_csv('../input/lish-moa/test_features.csv')

ss_svc = pd.read_csv('../input/lish-moa/sample_submission.csv')
ss_lr = ss_svc.copy()

cols = [c for c in ss_svc.columns.values if c != 'sig_id']


# In[4]:


def preprocess(df):
    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})
    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})
    del df['sig_id']
    return df

def log_loss_metric(y_true, y_pred):
    metrics = []
    for _target in train_targets.columns:
        metrics.append(log_loss(y_true.loc[:, _target], y_pred.loc[:, _target].astype(float), labels = [0,1]))
    return np.mean(metrics)

train = preprocess(train_features)
test = preprocess(test_features)

del train_targets['sig_id']


# In[5]:


top_feats = [  0,   1,   2,   3,   5,   6,   8,   9,  10,  11,  12,  14,  15,
        16,  18,  19,  20,  21,  23,  24,  25,  27,  28,  29,  30,  31,
        32,  33,  34,  35,  36,  37,  39,  40,  41,  42,  44,  45,  46,
        48,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,
        63,  64,  65,  66,  68,  69,  70,  71,  72,  73,  74,  75,  76,
        78,  79,  80,  81,  82,  83,  84,  86,  87,  88,  89,  90,  92,
        93,  94,  95,  96,  97,  99, 100, 101, 103, 104, 105, 106, 107,
       108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
       121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 132, 133, 134,
       135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
       149, 150, 151, 152, 153, 154, 155, 157, 159, 160, 161, 163, 164,
       165, 166, 167, 168, 169, 170, 172, 173, 175, 176, 177, 178, 180,
       181, 182, 183, 184, 186, 187, 188, 189, 190, 191, 192, 193, 195,
       197, 198, 199, 202, 203, 205, 206, 208, 209, 210, 211, 212, 213,
       214, 215, 218, 219, 220, 221, 222, 224, 225, 227, 228, 229, 230,
       231, 232, 233, 234, 236, 238, 239, 240, 241, 242, 243, 244, 245,
       246, 248, 249, 250, 251, 253, 254, 255, 256, 257, 258, 259, 260,
       261, 263, 265, 266, 268, 270, 271, 272, 273, 275, 276, 277, 279,
       282, 283, 286, 287, 288, 289, 290, 294, 295, 296, 297, 299, 300,
       301, 302, 303, 304, 305, 306, 308, 309, 310, 311, 312, 313, 315,
       316, 317, 320, 321, 322, 324, 325, 326, 327, 328, 329, 330, 331,
       332, 333, 334, 335, 338, 339, 340, 341, 343, 344, 345, 346, 347,
       349, 350, 351, 352, 353, 355, 356, 357, 358, 359, 360, 361, 362,
       363, 364, 365, 366, 368, 369, 370, 371, 372, 374, 375, 376, 377,
       378, 379, 380, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391,
       392, 393, 394, 395, 397, 398, 399, 400, 401, 403, 405, 406, 407,
       408, 410, 411, 412, 413, 414, 415, 417, 418, 419, 420, 421, 422,
       423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435,
       436, 437, 438, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450,
       452, 453, 454, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465,
       466, 468, 469, 471, 472, 473, 474, 475, 476, 477, 478, 479, 482,
       483, 485, 486, 487, 488, 489, 491, 492, 494, 495, 496, 500, 501,
       502, 503, 505, 506, 507, 509, 510, 511, 512, 513, 514, 516, 517,
       518, 519, 521, 523, 525, 526, 527, 528, 529, 530, 531, 532, 533,
       534, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547,
       549, 550, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563,
       564, 565, 566, 567, 569, 570, 571, 572, 573, 574, 575, 577, 580,
       581, 582, 583, 586, 587, 590, 591, 592, 593, 595, 596, 597, 598,
       599, 600, 601, 602, 603, 605, 607, 608, 609, 611, 612, 613, 614,
       615, 616, 617, 619, 622, 623, 625, 627, 630, 631, 632, 633, 634,
       635, 637, 638, 639, 642, 643, 644, 645, 646, 647, 649, 650, 651,
       652, 654, 655, 658, 659, 660, 661, 662, 663, 664, 666, 667, 668,
       669, 670, 672, 674, 675, 676, 677, 678, 680, 681, 682, 684, 685,
       686, 687, 688, 689, 691, 692, 694, 695, 696, 697, 699, 700, 701,
       702, 703, 704, 705, 707, 708, 709, 711, 712, 713, 714, 715, 716,
       717, 723, 725, 727, 728, 729, 730, 731, 732, 734, 736, 737, 738,
       739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751,
       752, 753, 754, 755, 756, 758, 759, 760, 761, 762, 763, 764, 765,
       766, 767, 769, 770, 771, 772, 774, 775, 780, 781, 782, 783, 784,
       785, 787, 788, 790, 793, 795, 797, 799, 800, 801, 805, 808, 809,
       811, 812, 813, 816, 819, 820, 821, 822, 823, 825, 826, 827, 829,
       831, 832, 833, 834, 835, 837, 838, 839, 840, 841, 842, 844, 845,
       846, 847, 848, 850, 851, 852, 854, 855, 856, 858, 860, 861, 862,
       864, 867, 868, 870, 871, 873, 874]

print((len(top_feats)))


# # CuML SVM Models

# In[6]:


scaler = StandardScaler()
X = scaler.fit_transform(train.values[:, top_feats])
x_tt = scaler.transform(test_features.values[:, top_feats])


# In[7]:


# from sklearn.svm import SVC, SVR
from cuml.svm import SVC, SVR

N_STARTS = 3
N_SPLITS = 5

res_svc = train_targets.copy()
ss_svc.loc[:, train_targets.columns] = 0
res_svc.loc[:, train_targets.columns] = 0

for tar in tqdm(list(range(train_targets.shape[1]))):

    start_time = time()
    targets = train_targets.values[:, tar]

    if targets.sum() >= N_SPLITS:

        for seed in range(N_STARTS):

            skf = StratifiedKFold(n_splits = N_SPLITS, random_state = seed, shuffle = True)

            for n, (tr, te) in enumerate(skf.split(targets, targets)):

                x_tr, x_val = X[tr], X[te]
                y_tr, y_val = targets[tr], targets[te]

                model = SVC(C = 40, cache_size = 2000)
                model.fit(x_tr, y_tr)
                ss_svc.loc[:, train_targets.columns[tar]] += model.decision_function(x_tt) / (N_SPLITS * N_STARTS)
                res_svc.loc[te, train_targets.columns[tar]] += model.decision_function(x_val) / N_STARTS

    score = log_loss(train_targets.loc[:, train_targets.columns[tar]], res_svc.loc[:, train_targets.columns[tar]])
    print((f'[{str(datetime.timedelta(seconds = time() - start_time))[2:7]}] SVM Target {tar}:', score))


# In[8]:


print(f'SVM OOF Metric: {log_loss_metric(train_targets, res_svc)}')
res_svc.loc[train['cp_type'] == 1, train_targets.columns] = 0
ss_svc.loc[test['cp_type'] == 1, train_targets.columns] = 0
print(f'SVM OOF Metric with postprocessing: {log_loss_metric(train_targets, res_svc)}')


# # Platt Scaling (Logistic Regression)

# In[9]:


X_new = res_svc[cols].values
x_tt_new = ss_svc[cols].values


# In[10]:


from sklearn.linear_model import LogisticRegression
# from cuml import LogisticRegression

N_STARTS = 3
N_SPLITS = 5

res_lr = train_targets.copy()
ss_lr.loc[:, train_targets.columns] = 0
res_lr.loc[:, train_targets.columns] = 0


for tar in tqdm(list(range(train_targets.shape[1]))):

    start_time = time()
    targets = train_targets.values[:, tar]

    if targets.sum() >= N_SPLITS:

        for seed in range(N_STARTS):

            skf = StratifiedKFold(n_splits = N_SPLITS, random_state = seed, shuffle = True)

            for n, (tr, te) in enumerate(skf.split(targets, targets)):

                x_tr, x_val = X_new[tr, tar].reshape(-1, 1), X_new[te, tar].reshape(-1, 1)
                y_tr, y_val = targets[tr], targets[te]

                model = LogisticRegression(C = 35, max_iter = 1000)
                model.fit(x_tr, y_tr)
                ss_lr.loc[:, train_targets.columns[tar]] += model.predict_proba(x_tt_new[:, tar].reshape(-1, 1))[:, 1] / (N_SPLITS * N_STARTS)
                res_lr.loc[te, train_targets.columns[tar]] += model.predict_proba(x_val)[:, 1] / N_STARTS

    score = log_loss(train_targets.loc[:, train_targets.columns[tar]], res_lr.loc[:, train_targets.columns[tar]])
    print((f'[{str(datetime.timedelta(seconds = time() - start_time))[2:7]}] LR Target {tar}:', score))


# In[11]:


print(f'LR OOF Metric: {log_loss_metric(train_targets, res_lr)}')
res_lr.loc[train['cp_type'] == 1, train_targets.columns] = 0
ss_lr.loc[test['cp_type'] == 1, train_targets.columns] = 0
print(f'LR OOF Metric with postprocessing: {log_loss_metric(train_targets, res_lr)}')


# # Submit

# In[12]:


ss_lr.to_csv('submission.csv', index = False)

