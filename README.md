### Zero-shot Dialog Generation

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

> 2. train model by Multilingual learning

```
 python run.py \
 	--corpus DailyDialog \
 	--do_train \
 	--hier TRUE \
 	--model hred \
 	--bidirectional TRUE \
 	
```
