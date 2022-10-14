#!/usr/bin/python
#coding:utf-8

import re
import sys
import numpy as np
import pandas as pd
import bz2
import gzip
import random
import time
from time import strftime, gmtime
from collections import defaultdict
import joblib
import metrics
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")

start_time = time.time()

input_file = sys.argv[1] #target VCF file
ImpRefIC_path = sys.argv[2] #software path
out_path = sys.argv[3] #output file path

#Basic information of the target VCF file
study_sample = []
dup_chrom = []
chrom = []
SNP_num = 0
with gzip.open(input_file, 'rt') as file3:
    for line in file3:
        if line.startswith("##"):
            pass
        elif line.startswith("#"):
            line_list = line.strip('\n').split('\t')
            study_sample.extend(line_list[i] for i in range(9, len(line_list)))
            continue
        else:
            SNP_num = SNP_num + 1
            line_list = line.strip('\n').split('\t')
            dup_chrom.append(line_list[0])
    for i in dup_chrom:
        if i not in chrom:
            chrom.append(i)

print("[INFO] Study samples: "+ str(len(study_sample)))
print("[INFO] The chromosomes contained in the target VCF file: " + ",".join(str(i) for i in chrom))
print("[INFO] Study snps: "+ str(SNP_num))

#Corresponding markers in the reference and target files must have identical CHROM, POS, REF, and ALT fields.
all_SNP = {}
consistent_SNP = defaultdict(list)
pair_dict = defaultdict(list)
with bz2.open(ImpRefIC_path + "/SNP.INFO.bz2",'rt') as file1:
    for line in file1:
        line_list = line.strip().split()
        all_SNP.update({line_list[0]+','+line_list[1]+','+line_list[2]+','+line_list[3]:line})
with gzip.open(input_file,'rt') as file2:
    for line in file2:
        if line.startswith("#"):
            pass
        else:
            line = line.replace('chr', '')
            line = line.replace('Chr', '')
            line_list = line.strip().split()
            lines = line_list[0]+','+line_list[1]+','+line_list[3]+','+line_list[4]
            if lines in all_SNP:
                consistent_SNP[lines].append(all_SNP[lines])

print("[INFO] Consistent snps: "+ str(len(consistent_SNP)))

#Determine whether the submitted VCF file has consistent SNPs with the reference panel.
if len(consistent_SNP) == 0:
    print("[INFO] The submitted VCF file does not have SNPs consistent with the reference panel, so the customized reference panel cannot be predicted.")
#Determine the number of consistent SNPs. If there are more than 50,000 SNPs, only 50,000 SNPs will be randomly selected to customize the reference panel.
if len(consistent_SNP) >= 50000:
    consistent_SNP_50K = random.sample(consistent_SNP.items(), 50000)
    for pair in consistent_SNP_50K:
        pair_list = list(pair)
        pair_key = pair_list[0]
        pair_value = ''.join(pair_list[1])
        pair_dict[pair_key].append(pair_value)
    consistent_SNP = pair_dict

print("[INFO] SNPs for classification and prediction: "+ str(len(consistent_SNP)) + "\n")

#Target VCF file
base = {'A': 0, 'T': 0.1, 'C': 0.3, 'G':0.7, 'N':0.9}
study_geno = {}
with gzip.open(input_file, 'rt') as file4:
    for line in file4:
        if line.startswith("#"):
            pass
        else:
            line = line.replace('chr', '')
            line = line.replace('Chr', '')
            var = line.strip('\n').split('\t', 6)[:5]
            if var[0]+","+var[1]+","+var[3]+","+var[4] in consistent_SNP:
                var = line.strip('\n').split('\t')
                allele = [var[3], var[4]]
                for i in range(9, len(var)):
                    if var[i].startswith('./.'):
                        var[i] = base['N']+base['N']
                    else:
                        var[i] = round((base[allele[int(re.split(r'[\|\/]', var[i])[0])]] + base[allele[int(re.split(r'[\|\/]', var[i])[1])]]),1)
                study_geno[var[0]+","+var[1]+","+var[3]+","+var[4]] = var[9:]

study_G = np.concatenate([[study_geno[i]] for i in consistent_SNP],axis=0)
study_G = study_G.T

#Reference file as model training set
ref_geno={}
with bz2.open(ImpRefIC_path + "/chr1-18.pos_snp_sample.matrix.bz2",'rt') as file5:
    for line in file5:
        line_list = line.strip().split()
        lines = (',').join(line_list[:4])
        if lines in consistent_SNP:
            ref_geno[lines] = line_list[4:]
ref_G = np.concatenate([[ref_geno[i]] for i in consistent_SNP],axis=0)

x = ref_G.T

#Label
ref_class = []
F = open(ImpRefIC_path + "/ref_class.txt",'rt')
for line in F:
    line=line.strip('\n')
    ref_class.append(int(line))

y = np.array(ref_class)

#Upsampling
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
x_resampled, y_resampled = ros.fit_resample(x, y)

#Model Initialization
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10, max_iter=1000, n_jobs=-1)

#Model Evaluation Metrics
def get_metrics(y_test, y_predicted):
    Accuracy = accuracy_score(y_test, y_predicted)
    Precision = precision_score(y_test, y_predicted, average='weighted')
    Recall = recall_score(y_test, y_predicted, average='weighted')
    F1 = f1_score(y_test, y_predicted, average='weighted')
    return Accuracy, Precision, Recall, F1

#Divide training set and test set
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size = 0.20, random_state=None)
print("[INFO] Successfully initialize a new model !")

#Model training
print("[INFO] Training the model …… ")
model.fit(x_train,y_train)
print("[INFO] Model training completed !\n")

#The trained model predicts the test set
y_pred=model.predict(x_test)
print("===================Confusion Matrix===================")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
Accuracy, Precision, Recall, F1 = get_metrics(y_test, y_pred)
print("Accuracy = %.4f" % Accuracy)
print("Precision = %.4f" % Precision)
print("Recall = %.4f" % Recall)
print("F1 = %.4f" % F1)

#Save model
joblib.dump(model, out_path + "/LogisticRegression.pkl")
print("[INFO] Model has been saved to" + out_path + "/LogisticRegression.pkl")

#The trained model predicts the target file
print("[INFO] The model starts predicting the target file …… ")
w_proba = model.predict_proba(study_G)
w = model.predict(study_G)

#Output predicted probabilities and customized reference population
ref_pop = []
f = open(ImpRefIC_path + "/ref_pop.txt",'rt')
for line in f:
    line_list = line.strip('\n').split()
    ref_pop.append(line_list)
population = np.array(ref_pop)

study_sample = np.array(study_sample).reshape(len(study_sample),1)
out_population = np.append(study_sample, population[w], axis=1)
np.savetxt(out_path + "/ImpRefIC.out.population", out_population, fmt='%s', delimiter='\t')
np.savetxt(out_path + "/ImpRefIC.out.ref.population", np.unique(population[w]), fmt='%s', delimiter='\t')
np.savetxt(out_path + "/ImpRefIC.out.population.proba", population[np.unique(y)].transpose(), fmt='%s', delimiter='\t')
file6 = open(out_path + "/ImpRefIC.out.population.proba", "ab")
np.savetxt(file6, w_proba, fmt='%.4f', delimiter='\t')

#Total running time
end_time = time.time()
run_time = end_time - start_time
run_time = strftime("%H:%M:%S", gmtime(run_time))

print("[INFO] Prediction complete! The predicted frequencies and predicted optimal reference population have been saved to " + out_path)
print("[INFO] Total time consumption is ",run_time)
print("\n")
print(chr(10059) + "Note: few consistent SNPs or insufficient chromosomal diversity will result in inaccurate predictions" + chr(10059))
