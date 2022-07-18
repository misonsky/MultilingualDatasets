#coding=utf-8
import argparse
import tensorflow as tf
import os
from utils.global_utils import StrategyObject
from trainer import Trainer
parser = argparse.ArgumentParser('parameters config for zero-shot generate conversation')
string_settings = parser.add_argument_group('string settings')
string_settings.add_argument('--data_dir',type=str,default="dataset",help='dataset path')
string_settings.add_argument('--gpu',type=str,default="0",help='which gpu device to use')
string_settings.add_argument('--corpus',type=str,default="visual",help='select task to train')
string_settings.add_argument('--lang',type=str,default="ru",help='which language to use')
string_settings.add_argument('--alignway',type=str,default="nalignment",help='the way of alignment')
string_settings.add_argument('--train_files',type=str,default="train.txt",help='train files')
string_settings.add_argument('--dev_files',type=str,default="dev.txt",help='the dev file evaluating the model')
string_settings.add_argument('--test_files',type=str,default="test.txt",help='the test file')
string_settings.add_argument('--train_record',type=str,default="train.%s_tfrecord",help='the train tfrecorder file')
string_settings.add_argument('--dev_record',type=str,default="dev.%s_tfrecord",help='the dev tfrecorder file')
string_settings.add_argument('--test_record',type=str,default="test.%s_tfrecord",help='the test tfrecorder file')
string_settings.add_argument('--model',type=str,default="vhred",help='seq2seq/transformer/fairseq')
string_settings.add_argument('--model_dir',type=str,default="TrainModel",help='path to save model')
string_settings.add_argument('--vocab',type=str,default="vocabs",help="the orign vocab path")
string_settings.add_argument('--processDir',type=str,default="processdir",help='save the pretrain Model')
string_settings.add_argument('--rnn_type',type=str,default="gru",help='encoder type')
string_settings.add_argument('--embedding',type=str,default="embeddings",help="golve word2 vector")
string_settings.add_argument('--EvalEmb',type=str,default="EvaluationEmbedding",help="golve word2 vector")

boolean_settings = parser.add_argument_group('boolean settings')
boolean_settings.add_argument('--do_lower_case',type=bool,default=True,help='whether employ lower case')
boolean_settings.add_argument('--load_last_ckpt',type=bool,default=False,help='whether training the model from the last checkpoint')
boolean_settings.add_argument('--no_cuda',type=bool,default=False,help='whether use the cuda device')
boolean_settings.add_argument('--do_prepare',type=bool,default=False,help='prepare the dataset for supervised training')
boolean_settings.add_argument('--do_train',type=bool,default=False,help='train the supervised model')
boolean_settings.add_argument('--do_eval',type=bool,default=True,help='Whether to run eval on the dev set.')
boolean_settings.add_argument('--do_predict',type=bool,default=False,help='prediction the result for supervised')
boolean_settings.add_argument('--fp16',type=bool,default=False,help='using the mixed_float16 when traing')
boolean_settings.add_argument('--drop_last',type=bool,default=False,help='whether drop the last dataset')
boolean_settings.add_argument('--teach_force',type=bool,default=True,help="using teach force when decoder stage")
boolean_settings.add_argument('--hier',type=bool,default=True,help="whether using hier structure")
boolean_settings.add_argument('--graphModel',type=bool,default=False,help="whether using graph model structure")
boolean_settings.add_argument('--bidirectional',type=bool,default=True,help="whether using bidirectional rnn")
boolean_settings.add_argument('--bow',type=bool,default=False,help="bow loss")

scaler_settings = parser.add_argument_group('scaler settings')
scaler_settings.add_argument('--max_turn',type=int,default=10,help='max number turn of conversation')
scaler_settings.add_argument('--max_utterance_len',type=int,default=50,help='max length of utterance')
scaler_settings.add_argument('--decode_length',type=int,default=30,help='length for decoder')
scaler_settings.add_argument('--beam_size',type=int,default=6,help="beam size of decoder")
scaler_settings.add_argument('--eval_steps',type=int,default=1000,help='number steps eval the model')
scaler_settings.add_argument('--log_steps',type=int,default=50,help='number steps log info')
scaler_settings.add_argument('--num_layers',type=int,default=2,help='the number of encoder layers')
scaler_settings.add_argument('--v2v_layers',type=int,default=1,help="the number of encoder layers")
scaler_settings.add_argument('--decoder_layers',type=int,default=1,help='the number of encoder layers')
scaler_settings.add_argument('--num_heads',type=int,default=8,help='head number of multi-head attention')
scaler_settings.add_argument('--kl_annealing_iter',type=int,default=2500,help="kl_annealing_iter")
scaler_settings.add_argument('--gradient_accumulation_steps',type=int,default=2,help='gradient accumulation steps')
scaler_settings.add_argument('--train_batch_size',type=int,default=100,help='Batch size per GPU/TPU core/CPU for training.')
scaler_settings.add_argument('--eval_batch_size',type=int,default=100,help='Batch size per GPU/TPU core/CPU for evaluation.')
scaler_settings.add_argument('--d_model',type=int,default=512,help='the hidden size of model')
scaler_settings.add_argument('--dff',type=int,default=1024,help='the hidden size of model')
scaler_settings.add_argument('--emb_size',type=int,default=512,help='the embedding dimension')
scaler_settings.add_argument('--warmup_steps',type=int,default=1000,help='Linear warmup over warmup_steps.')
scaler_settings.add_argument('--num_train_epochs',type=int,default=500,help='Total number of training epochs to perform.')
scaler_settings.add_argument('--negatives_num',type=int,default=50,help='the negatives number when training.')
scaler_settings.add_argument('--dropout',type=float,default=0.0,help='dropout rate')
scaler_settings.add_argument('--learning_rate',type=float,default=0.0001,help='The initial learning rate for Adam.')
scaler_settings.add_argument('--weight_decay',type=float,default=0,help='Weight decay if we apply some.')
scaler_settings.add_argument('--adam_beta1',type=float,default=0.9,help='Beta1 for Adam optimizer')
scaler_settings.add_argument('--adam_beta2',type=float,default=0.999,help='Beta2 for Adam optimizer')
scaler_settings.add_argument('--adam_epsilon',type=float,default=1e-8,help='Epsilon for Adam optimizer.')
scaler_settings.add_argument('--max_grad_norm',type=float,default=1.0,help='Max gradient norm.')
scaler_settings.add_argument('--temperature',type=float,default=0.03,help='the temperature value when contrastive learning')
scaler_settings.add_argument('--alpha',type=float,default=0.6,help="alpha for length penalty")
scaler_settings.add_argument('--en_hidden_list',type=list,default=[512,512,512],help="fairseq hidden list")
scaler_settings.add_argument('--en_kwidths_list',type=list,default=[3,3,3],help="keral size of fairseq")
scaler_settings.add_argument('--de_hidden_list',type=list,default=[512],help="fairseq hidden list")
scaler_settings.add_argument('--de_kwidths_list',type=list,default=[3],help="keral size of fairseq")
config=parser.parse_args()
strategy = StrategyObject(config)
def makdDirs():
    languages = ["en",'de',"it","es","zh","fr","ru","demo"]
    for lan in languages:
        cpath = os.path.join(config.processDir,
                             config.corpus,
                             config.alignway,
                             lan)
        if not os.path.exists(cpath):
            os.makedirs(cpath)
        mpath = os.path.join(config.model_dir,
                             config.corpus,
                             config.alignway,
                             lan,
                             config.model)
        if not os.path.exists(mpath):
            os.makedirs(mpath)
        epath = os.path.join(config.embedding,
                             config.corpus,
                             config.alignway,
                             lan)
        if not os.path.exists(epath):
            os.makedirs(epath)
def run():
    if config.fp16:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
        tf.keras.mixed_precision.experimental.set_policy(policy) 
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    trainer_handle = Trainer(strategy)
    makdDirs()
    if config.do_prepare:
        trainer_handle.prepare()
    elif config.do_train:
        trainer_handle.train()
    elif config.do_eval:
        metrics_result = trainer_handle.evaluate(datasetInstance=None,only_evaluation=True)
        print(metrics_result)
    elif config.do_predict:
        trainer_handle.predict(datasetInstance=None)
if __name__=="__main__":
    run()
