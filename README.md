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
|  表头   | 表头  |
|  ----  | ----  |
| 单元格  | 单元格 |
| 单元格  | 单元格 |
```

# 1 标题
粗略的标题划分，可以通过“=”和“-”实现。

在文字下方加上“=”，“=”上方的文字就变成了大标题了，等于号的个数无限制，但一定大于0。
比如：

```
标题A
===
```

效果如下：

标题A
===

文字下方加上“-”，上方的文章变成了中标题，比如:

```
标题A
---
```

效果如下：

标题B
---

更精细的标题划分，可以通过“#”来实现，示例如下：
```
# 一级标题
## 二级标题
### 三级标题
#### 四级标题
##### 五级标题
###### 六级标题
```

效果如下：

# 一级标题
## 二级标题
### 三级标题
#### 四级标题
##### 五级标题
###### 六级标题

# 2 字体

字体可以实现加粗、斜体、删除线效果，示例如下：

```
**这是加粗的文字**
*这是倾斜的文字*`
***这是斜体加粗的文字***
~~这是加删除线的文字~~
```
实际效果为：

**这是加粗的文字**

*这是倾斜的文字*`

***这是斜体加粗的文字***

~~这是加删除线的文字~~

# 3 列表与表格

列表有有序列表和无序列表，无序列表可用“*”或“-”作为前导符号，示例如下：
```
* 无序列表1
- 无序列表2
* 无序列表3
- 无序列表4
```

实际效果为：

* 无序列表1
- 无序列表2

有序列表则可以使用“1. ”或“1) ”作为前导符号，示例如下：
```
1. 有序列表1
2) 有序列表2
```

实际效果为：

1. 有序列表1
2) 有序列表2

表格的示例用法如下：
```
|  表头   | 表头  |
|  ----  | ----  |
| 单元格  | 单元格 |
| 单元格  | 单元格 |
```

效果如下：

|  表头   | 表头  |
|  ----  | ----  |
| 单元格  | 单元格 |
| 单元格  | 单元格 |

表格内的不同文字对齐方式：
1) -: 设置内容和标题栏居右对齐。
2) :- 设置内容和标题栏居左对齐。
3) :-: 设置内容和标题栏居中对齐。

也可参考我另外一篇博客：[CSDN-Markdown编写博客](https://blog.csdn.net/luobing4365/article/details/113106959)

# 4 分割线与引用

分割线可用来隔开内容，三个或三个以上的“-”或“*”都可以，示例如下：

```
---
***
```

效果如下：

---
***

引用的效果，在平常玩微信的时候经常看的。可以通过1个或者多个“>”来实现，示例如下：
```
> 引用内容1
>> 引用内容2
>>> 引用内容3
```

效果为：

> 引用内容1
>> 引用内容2
>>> 引用内容3

# 5 图片

图片的格式为：
```
![图片说明](图片地址"图片标题")
```

图片可以使用外链接，只要找到可以存放的外链地址就可以了。在Gitee中，我一般会在同级目录下建立一个文件夹，专门用来存放需要显示的图片。

上传以后，在Gitee网站打开图片，点击“原始数据”，得到其地址。注意，直接点开的图片地址不行，一定是原始数据的地址才可以。

另外一种方法，在Gitee网站上，提供了插入图片的功能。打开Markdown文件，选择“编辑”，进入编辑状态，可以看到插入图片的功能键。以下为插入后的图片示例。

![示例图片](https://images.gitee.com/uploads/images/2021/0827/162908_16dc411a_791211.png "youli.png")

由于我的仓库，很多时候也需要上传到Github上，所以习惯采用第一种方法，比如UEFI开发探索的系列博客，其仓库的Readme.md，显示图片就在同级目录mygiteepic下：
[UEFI开发探索的仓库说明](https://gitee.com/luobing4365/uefi-explorer/blob/master/README.md)

有个小技巧，可以通过图片链接网址。首先在本文编辑中，我多次使用了文本链接，其格式为：
```
[链接地址说明](链接地址)
```

可以采用同样的方式来进行，格式为：

```
[![图片说明](图片地址"图片标题")](链接地址)
```

将上述图片链接到我的博客地址：<br>
[![示例图片](https://images.gitee.com/uploads/images/2021/0827/162908_16dc411a_791211.png "youli.png")](https://blog.csdn.net/luobing4365)

点击图片，就会跳转到博客地址了。

# 6 代码

对于单行的代码，可以使用“`”进行包含，代码块则可以使用“```”进行包含，个人更习惯于用后者。

示例如下：
```
 `hello,world!'
```

效果为：

`hello,world!`

```
  \```  使用时请将反斜杆去除
        这是为了演示，只能加上反斜杠
  \```
```

效果为:
```  使用时请将反斜杆去除
     这是为了演示，只能加上反斜杠
```

# 7 Emoji

Gitee平台上还提供了Emoji图案，有兴趣可以用用。
比如：

`:heart: ` :heart:  <br>
`:sparkles: ` :sparkles: <br>
`:+1: ` :+1: <br>
`:rooster: ` :rooster: <br>

# 8 HTML语法

有些效果，使用markdown的语法，很难实现，这时就要用到HTML语法了。

不过，不同的平台，对HTM语法的支持并不一致。目前为止，只找到使用HTML语法做表格的例子，如下：

<table>
  <tr>
    <th>Host Type</th>
    <th>Toolchain</th>
    <th>Branch</th>
    <th>Build Status</th>
    <th>Test Status</th>
    <th>Code Coverage</th>
  </tr>
  <tr>
    <td>Windows</td>
    <td>VS2019</td>
    <td>master</td>
    <td>
      <a  href="https://dev.azure.com/tianocore/edk2-ci/_build/latest?definitionId=32&branchName=master">
      <img src="https://dev.azure.com/tianocore/edk2-ci/_apis/build/status/Windows%20VS2019%20CI?branchName=master"/></a>
    </td>
    <td>
      <a  href="https://dev.azure.com/tianocore/edk2-ci/_build/latest?definitionId=32&branchName=master">
      <img src="https://img.shields.io/azure-devops/tests/tianocore/edk2-ci/32.svg"/></a>
    </td>
    <td>
      <a  href="https://dev.azure.com/tianocore/edk2-ci/_build/latest?definitionId=32&branchName=master">
      <img src="https://img.shields.io/badge/coverage-coming_soon-blue"/></a>
    </td>
  </tr>
  <tr>
    <td>Ubuntu</td>
    <td>GCC</td>
    <td>master</td>
    <td>
      <a  href="https://dev.azure.com/tianocore/edk2-ci/_build/latest?definitionId=31&branchName=master">
      <img src="https://dev.azure.com/tianocore/edk2-ci/_apis/build/status/Ubuntu%20GCC5%20CI?branchName=master"/></a>
    </td>
    <td>
      <a  href="https://dev.azure.com/tianocore/edk2-ci/_build/latest?definitionId=31&branchName=master">
      <img src="https://img.shields.io/azure-devops/tests/tianocore/edk2-ci/31.svg"/></a>
    </td>
    <td>
      <a  href="https://dev.azure.com/tianocore/edk2-ci/_build/latest?definitionId=31&branchName=master">
      <img src="https://img.shields.io/badge/coverage-coming_soon-blue"/></a>
    </td>
  </tr>
</table>

# 9 使用Gitee的编辑工具

虽然Markdown的语法比较简单，但有时候记住这些规则还是比较麻烦的。很多时候，可以直接使用Gitee提供的编辑工具。

Gitee中提供了“编辑”和“Web IDE”，很容易上手。

这篇文档，就是使用Web IDE完成的。


