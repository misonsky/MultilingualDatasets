### 522Zero-shot Dialog Generation

### Datasets
The multilingual versions datasets of DailyDialog and DSTC7, which includes seven languages: English, Chinese, German, Russian, Spanish, French and Italian.
### Models
We provide hred, vhcr, vhred, hran, transformer and HTransformer Baselines, our manuscript only provides the experimental results of hred, vhred, transformer and HTransformer.

#### This code using the MBERT Tokenizer  

### How to run

> 1. prepare the datasets

```
 python run.py \
 	--corpus DailyDialog \
 	--do_prepare
```

> 2. train model by Multilingual learning(specify the model parameter to switch between different models)

```
 python run.py \
 	--corpus DailyDialog \
 	--do_train \
 	--hier true \
 	--model hred \
 	--bidirectional true \
 	
```

> 3. evaluate  
```
python run.py \
 	--corpus DailyDialog \
 	--do_eval \
 	--hier true \
 	--model hred \
 	--bidirectional true \
```

###  T-Test

1. The p-value of models on Germen.

|       | HRED  | VHRED | Trans | HTrans |
| ----- | ----- | ----- | ----- | ------ |
| Daily | 0.054 | 0.062 | 0.054 | 0.175  |
| DSTC7 | 0.193 | 0.053 | 0.057 | 0.057  |

2. The p-value of models on Spanish.  

|       | HRED  | VHRED | Trans | HTrans |
| ----- | ----- | ----- | ----- | ------ |
| Daily | 0.176 | 0.29  | 0.747 | 0.127  |
| DSTC7 | 0.216 | 0.12  | 0.975 | 0.0576 |

3. The p-value of models on French.  

|       | HRED | VHRED | Trans | HTrans |
| ----- | ---- | ----- | ----- | ------ |
| Daily | 0.63 | 0.158 | 0.374 | 0.159  |
| DSTC7 | 0.92 | 0.056 | 0.123 | 0.455  |

4. The p-value of models on Italian.

|       | HRED  | VHRED | Trans | HTrans |
| ----- | ----- | ----- | ----- | ------ |
| Daily | 0.058 | 0.072 | 0.087 | 0.672  |
| DSTC7 | 0.093 | 0.045 | 0.051 | 0.076  |

5. The p-value of models on Russian.

|       | HRED  | VHRED | Trans | HTrans |
| ----- | ----- | ----- | ----- | ------ |
| Daily | 0.082 | 0.065 | 0.052 | 0.056  |
| DSTC7 | 0.491 | 0.321 | 0.12  | 0.068  |

6. The p-value of models on Chinese.

|       | HRED  | VHRED | Trans | HTrans |
| ----- | ----- | ----- | ----- | ------ |
| Daily | 0.068 | 0.075 | 0.054 | 0.057  |
| DSTC7 | 0.152 | 0.057 | 0.066 | 0.073  |


###  Case Study  

![Case study](case.png)  

An case study is provided in Table 9  to demonstrate the values of augmented data.  We can observe that the responses of HRED and VHRD contain context-independent information without using data augmentation. Specifically, "like my experience and i have a good idea" in HRED and "i like sports" in VHRED are context independent. The responses generated by HRED and VHRED are more informative and mroe coherent when using data augmentation. Models can utilize cross linguistic knowledge to generate more informative and coherent responses by multilingual data augmentation.  











