# ImpRef

## Contents
* ### Overview
* ### Installation
* ### Running
* ### Input
* ### Output
* ### Citation

## Overview
ImpRef is an intelligent method for customizing reference panels prior to genotype imputation. Based on the logistic regression algorithm, the method is trained to classify the SNP data of 64 breeds/lines of pigs around the world, and combine the breeds/lines most similar to the target sample sequence as a customized reference panel, achieving higher imputation accuracy, especially for low frequency and rare variants.

## Installation
```
git clone --recursive https://github.com/klzhang2022/ImpRef.git
```
### ✳Requirements
* python 3  (https://www.python.org)
* python modules and packages
```
import re
import sys
import numpy as np
import pandas as pd
import bz2
import gzip
import time
from time import strftime, gmtime
from collections import defaultdict
import joblib
import metrics
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
```

## Running
```
cd ImpRef_path   #ImpRef installation directory

python3 ImpRef.py /example/test.vcf.gz ./ /example
```
```

/example/test.vcf.gz   #Compressed VCF file of the target sample

./                     #ImpRef installation directory

/example               #Output path
```

## Input
Compressed VCF file (https://samtools.github.io/hts-specs/VCFv4.2.pdf)
```
##fileformat=VCFv4.2
##fileDate=20090805
##source=myImputationProgramV3.1
##reference=file:///seq/references/1000GenomesPilot-NCBI36.fasta
##contig=<ID=20,length=62435964,assembly=B36,md5=f126cdf8a6e0c7f379d618ff66beb2da,species="Homo sapiens",taxonomy=x>
##phasing=partial
##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of Samples With Data">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
##INFO=<ID=AA,Number=1,Type=String,Description="Ancestral Allele">
##INFO=<ID=DB,Number=0,Type=Flag,Description="dbSNP membership, build 129">
##INFO=<ID=H2,Number=0,Type=Flag,Description="HapMap2 membership">
##FILTER=<ID=q10,Description="Quality below 10">
##FILTER=<ID=s50,Description="Less than 50% of samples have data">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
##FORMAT=<ID=HQ,Number=2,Type=Integer,Description="Haplotype Quality">
#CHROM POS ID REF ALT QUAL FILTER INFO FORMAT NA00001 NA00002 NA00003
20 14370 rs6054257 G A 29 PASS NS=3;DP=14;AF=0.5;DB;H2 GT:GQ:DP:HQ 0|0:48:1:51,51 1|0:48:8:51,51 1/1:43:5:.,.
20 17330 . T A 3 q10 NS=3;DP=11;AF=0.017 GT:GQ:DP:HQ 0|0:49:3:58,50 0|1:3:5:65,3 0/0:41:3
20 1110696 rs6040355 A G,T 67 PASS NS=2;DP=10;AF=0.333,0.667;AA=T;DB GT:GQ:DP:HQ 1|2:21:6:23,27 2|1:2:0:18,2 2/2:35:4
20 1230237 . T . 47 PASS NS=3;DP=13;AA=T GT:GQ:DP:HQ 0|0:54:7:56,60 0|0:48:4:51,51 0/0:61:2
20 1234567 microsat1 GTC G,GTCT 50 PASS NS=3;DP=9;AA=G GT:GQ:DP 0/1:35:4 0/2:17:2 1/1:40:3
```

## Output
* ImpRef.out.population.proba

* ImpRef.out.population

* ImpRef.out.ref.population

```
#ImpRef.out.population.prob

probability matrix of target samples and 64 breeds/lines
```
|  American_Yorkshire  |  Canadian_Yorkshire  |  Danish_Yorkshire  |  Dutch_Yorkshire  |  French_Yorkshire  |  Unknown_Yorkshire_lines  |  Landrace  |  Duroc  |  Berkshire  |  Goettingen_Minipig  |  Hampshire  |  Iberian  |  Mangalica  |  Pietrain  |  Angler_Sattleschwein  |  British_Saddleback  |  Bunte_Bentheimer  |  Calabrese  |  Casertana  |  Chato_Murciano  |  Cinta_Senese  |  Gloucester_Old_Spot  |  Large_Black  |  Leicoma  |  Linderodsvin  |  Middle_White  |  Nero_Siciliano  |  Tamworth  |  European_Wild_boar  |  Yucatan_minipig  |  Creole  |  American_Wild_boar  |  Bamei  |  Baoshan  |  Enshi_black  |  Erhualian  |  Hetao  |  Jinhua  |  Korean_black_pig  |  Laiwu  |  Meishan  |  Min  |  Neijiang  |  Rongchang  |  Tibetan  |  Tongcheng  |  Hubei_White  |  Daweizi  |  Jiangquhai  |  Leping_Spotted  |  Penzhou  |  songliao_black_pig  |  Taihu  |  Wannan_Spotted  |  Wujin  |  Ya_nan  |  Diannanxiaoer  |  Luchuan  |  Wuzhishan  |  Bamaxiang  |  MiniLEWE  |  Xiang  |  Asia_Wild_boar  |  Hybrid  |
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|  0.0000  |  0.0000  |  0.9999  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |
|  0.0000  |  0.0000  |  1.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |
|  0.0000  |  0.0000  |  0.9999  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |
|  0.0000  |  0.0000  |  0.9999  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |
|  0.0000  |  0.0000  |  0.9999  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |
|  0.9988  |  0.0002  |  0.0002  |  0.0001  |  0.0001  |  0.0001  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0002  |
|  0.9962  |  0.0003  |  0.0002  |  0.0001  |  0.0002  |  0.0007  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0000  |  0.0018  |
|  0.0253  |  0.0036  |  0.0564  |  0.0032  |  0.0043  |  0.0644  |  0.1547  |  0.0002  |  0.0017  |  0.0010  |  0.0018  |  0.0010  |  0.0027  |  0.0022  |  0.0051  |  0.0016  |  0.0026  |  0.0012  |  0.0015  |  0.0026  |  0.0009  |  0.0009  |  0.0010  |  0.0018  |  0.0029  |  0.0027  |  0.0077  |  0.0009  |  0.0036  |  0.0010  |  0.0020  |  0.0010  |  0.0006  |  0.0003  |  0.0013  |  0.0000  |  0.0012  |  0.0001  |  0.0013  |  0.0007  |  0.0002  |  0.0037  |  0.0001  |  0.0000  |  0.0009  |  0.0002  |  0.0259  |  0.0002  |  0.0003  |  0.0003  |  0.0022  |  0.0015  |  0.0013  |  0.0003  |  0.0005  |  0.0001  |  0.0002  |  0.0001  |  0.0003  |  0.0001  |  0.0006  |  0.0002  |  0.0006  |  0.5914  |
|  0.0134  |  0.0171  |  0.1671  |  0.0033  |  0.0057  |  0.0639  |  0.2500  |  0.0002  |  0.0019  |  0.0017  |  0.0012  |  0.0006  |  0.0016  |  0.0029  |  0.0054  |  0.0015  |  0.0012  |  0.0012  |  0.0016  |  0.0022  |  0.0013  |  0.0009  |  0.0013  |  0.0027  |  0.0023  |  0.0035  |  0.0030  |  0.0014  |  0.0031  |  0.0019  |  0.0022  |  0.0013  |  0.0012  |  0.0005  |  0.0042  |  0.0001  |  0.0018  |  0.0001  |  0.0011  |  0.0008  |  0.0005  |  0.0026  |  0.0001  |  0.0001  |  0.0013  |  0.0005  |  0.0641  |  0.0004  |  0.0004  |  0.0004  |  0.0038  |  0.0017  |  0.0022  |  0.0006  |  0.0005  |  0.0002  |  0.0004  |  0.0001  |  0.0009  |  0.0005  |  0.0012  |  0.0003  |  0.0003  |  0.3389  |
|  0.0209  |  0.0093  |  0.0347  |  0.0013  |  0.0028  |  0.0273  |  0.0732  |  0.0002  |  0.0021  |  0.0013  |  0.0012  |  0.0005  |  0.0017  |  0.0015  |  0.0026  |  0.0011  |  0.0023  |  0.0008  |  0.0011  |  0.0018  |  0.0010  |  0.0009  |  0.0006  |  0.0014  |  0.0019  |  0.0020  |  0.0031  |  0.0007  |  0.0014  |  0.0009  |  0.0016  |  0.0006  |  0.0003  |  0.0002  |  0.0017  |  0.0000  |  0.0011  |  0.0001  |  0.0007  |  0.0007  |  0.0002  |  0.0035  |  0.0001  |  0.0001  |  0.0007  |  0.0002  |  0.0247  |  0.0003  |  0.0003  |  0.0003  |  0.0021  |  0.0009  |  0.0018  |  0.0002  |  0.0004  |  0.0001  |  0.0002  |  0.0000  |  0.0002  |  0.0003  |  0.0006  |  0.0002  |  0.0004  |  0.7536  |
