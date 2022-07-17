#coding=utf-8

import os
import json
import datetime
import math
from tqdm import tqdm
import numpy as np
import pickle as pkl
from DataSetBeam import DataSetBeam
from tensorflow.python.distribute.values import PerReplica
from utils.optimization import create_optimizer,GradientAccumulator
from model.hred import Seq2Seq, HRED
from model.Atthred import AttSeq2Seq,AttHRED
from model.transformer import HierTransformer,ConTransformer
from model.vhred import VHRED,VSeq2Seq
from model.vhcr import VHCR
from model.dshred import DSHRED
from model.hran import HRAN
import tensorflow as tf
from glob import glob
from metrics.evaluationtool import cal_greedy_matching_matrix,cal_embedding_average,cal_vector_extrema
from metrics.evaluationtool import cal_Distinct,compute_bleu_rouge_single_prediction,cal_corpus_bleu
def validation_parameters(config):
    if not config.do_prepare and not config.do_train and not config.do_eval and not config.do_predict:
        raise ValueError("must specify one of them")
    model_dir = os.path.join(config.model_dir,config.corpus,config.model)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    process_dir = os.path.join(config.processDir,config.corpus)
    if not os.path.exists(process_dir):
        os.makedirs(process_dir)

def combine_features(features):
    mul_features = {}
    for FeatureName,valueFeature in features.items():
        if FeatureName not in mul_features:
            mul_features[FeatureName] = valueFeature
        else:
            mul_features[FeatureName] = tf.concat([valueFeature,mul_features[FeatureName]],axis=0)
    return mul_features
#register model
def validation_hierModel_parameters(config):
    HierModel = ["hred","vhred","hiertransformer","atthred","hierfairseq","vhcr","dshred","hran","v2vhred"]
    if config.hier:
        assert config.model in HierModel
    else:
        assert config.model not in HierModel
def get_model(config,vocab_size,embedding_matrix,PROM,EOS,PAD):
    validation_hierModel_parameters(config)
    if config.model == "hred" and config.hier:
        net = HRED(vocab_size = vocab_size,
                embedding_dim = config.emb_size,
                matrix = embedding_matrix,
                config = config,
                PROM=PROM,
                EOS=EOS,
                PAD=PAD,
                PROM=PROM)
    elif config.model == "seq2seq" and not config.hier:
        net = Seq2Seq(vocab_size = vocab_size,
                embedding_dim = config.emb_size,
                matrix = embedding_matrix,
                config = config,
                PROM=PROM,
                EOS=EOS,
                PAD=PAD)
    elif config.model == "atthred" and config.hier:
        net = AttHRED(vocab_size = vocab_size,
                embedding_dim = config.emb_size,
                matrix = embedding_matrix,
                config = config,
                PROM=PROM,
                EOS=EOS,
                PAD=PAD)
    elif config.model == "attseq2seq" and not config.hier:
        net = AttSeq2Seq(vocab_size = vocab_size,
                embedding_dim = config.emb_size,
                matrix = embedding_matrix,
                config = config,
                PROM=PROM,
                EOS=EOS,
                PAD=PAD,)
    elif config.model == "vhred" and config.hier:
        net = VHRED(vocab_size = vocab_size,
                embedding_dim = config.emb_size,
                matrix = embedding_matrix,
                config = config,
                PROM=PROM,
                EOS=EOS,
                PAD=PAD,)
    elif config.model == "vseq2seq":
        net = VSeq2Seq(vocab_size = vocab_size,
                embedding_dim = config.emb_size,
                matrix = embedding_matrix,
                config = config,
                PROM=PROM,
                EOS=EOS,
                PAD=PAD,)
    elif config.model == "vhcr" and config.hier:
        net = VHCR(vocab_size = vocab_size,
                embedding_dim = config.emb_size,
                matrix = embedding_matrix,
                config = config,
                PROM=PROM,
                EOS=EOS,
                PAD=PAD,)
    elif config.model == "dshred" and config.hier:
        net = DSHRED(vocab_size = vocab_size,
                embedding_dim = config.emb_size,
                matrix = embedding_matrix,
                config = config,
                PROM=PROM,
                EOS=EOS,
                PAD=PAD,)
    elif config.model == "hran" and config.hier:
        net = HRAN(vocab_size = vocab_size,
                embedding_dim = config.emb_size,
                matrix = embedding_matrix,
                config = config,
                PROM=PROM,
                EOS=EOS,
                PAD=PAD,)
    elif config.model == "hiertransformer" and config.hier:
        net = HierTransformer(vocab_size = vocab_size,
                    matrix = embedding_matrix,
                    config = config,
                    PROM=PROM,
                    EOS=EOS,
                    PAD=PAD,
                    rate=0.1)
    elif config.model == "transformer":
        net = ConTransformer(vocab_size = vocab_size,
                    matrix = embedding_matrix,
                    config = config,
                    PROM=PROM,
                    EOS=EOS,
                    PAD=PAD,
                    rate=0.1)
    tf.print(" the select model is {}".format(net))
    return net
class Trainer(object):
    def __init__(self,strategy,model=None,optimizer=None,LRSchedule=None):
        self.lr_scheduler=LRSchedule
        self.strategy = strategy
        self.model=model
        self.optimizer=optimizer
        self.gradient_accumulator = GradientAccumulator()
        self.global_step = 0
        self.meta_dict = None
        self.best_metrics = float("inf")
        validation_parameters(strategy.config)
    def run_model(self, features,prediction):
        """
            parameters:
                features:dict()
                return:
                    sup: loss, logits, prediction_label
                    contra: loss, logits, real label
        """
        results_dict={}
        training = not prediction
        if self.strategy.config.model =="seq2seq":
            outputs,loss=self.model(features,training=training)
        elif self.strategy.config.model =="v2vhred" and self.strategy.config.hier:
            outputs,loss=self.model(features,training=training)
        elif self.strategy.config.model == "vseq2seq":
            outputs,loss=self.model(features,training=training)
        elif self.strategy.config.model =="hred" and self.strategy.config.hier:
            outputs,loss=self.model(features,training=training)
        elif self.strategy.config.model =="attseq2seq":
            outputs,loss=self.model(features,training=training)
        elif self.strategy.config.model =="atthred" and self.strategy.config.hier:
            outputs,loss=self.model(features,training=training)
        elif self.strategy.config.model == "vhred" and self.strategy.config.hier:
            outputs,loss=self.model(features,training=training)
        elif self.strategy.config.model == "vhcr" and self.strategy.config.hier:
            outputs,loss=self.model(features,training=training)
        elif self.strategy.config.model == "dshred" and self.strategy.config.hier:
            outputs,loss=self.model(features,training=training)
        elif self.strategy.config.model == "hran" and self.strategy.config.hier:
            outputs,loss=self.model(features,training=training)
        elif self.strategy.config.model =="hiertransformer" and self.strategy.config.hier:
            outputs,loss=self.model(features,training = training)
        elif self.strategy.config.model =="transformer":
            outputs,loss=self.model(features,training = training)
        elif self.strategy.config.model == "hierfairseq" and self.strategy.config.hier:
            outputs,loss=self.model(features,training=training)
        elif self.strategy.config.model == "fairseq":
            outputs,loss=self.model(features,training=training)
        results_dict["outputs"] = outputs
        results_dict["loss"] = loss
        if prediction:
            if self.strategy.config.model =="seq2seq":
                outputs = self.model.BeamDecoder(features)
            elif self.strategy.config.model == "vseq2seq":
                outputs = self.model.BeamDecoder(features)
            elif self.strategy.config.model == "v2vhred" and self.strategy.config.hier:
                outputs = self.model.BeamDecoder(features)
            elif self.strategy.config.model =="hred" and self.strategy.config.hier:
                outputs = self.model.BeamDecoder(features)
            elif self.strategy.config.model =="attseq2seq":
                outputs = self.model.BeamDecoder(features)
            elif self.strategy.config.model =="atthred" and self.strategy.config.hier:
                outputs = self.model.BeamDecoder(features)
            elif self.strategy.config.model == "vhcr" and self.strategy.config.hier:
                outputs = self.model.BeamDecoder(features)
            elif self.strategy.config.model =="hiertransformer" and self.strategy.config.hier:
                outputs = self.model.BeamDecoder(features,training = False)
            elif self.strategy.config.model == "vhred" and self.strategy.config.hier:
                outputs = self.model.BeamDecoder(features)
            elif self.strategy.config.model == "dsred" and self.strategy.config.hier:
                outputs = self.model.BeamDecoder(features)
            elif self.strategy.config.model == "hran" and self.strategy.config.hier:
                outputs = self.model.BeamDecoder(features)
            elif self.strategy.config.model =="transformer":
                outputs = self.model.BeamDecoder(features,training = False)
            elif self.strategy.config.model =="hierfairseq" and self.strategy.config.hier:
                outputs = self.model.BeamDecoder(features,training = False)
            elif self.strategy.config.model == "fairseq":
                outputs = self.model.BeamDecoder(features,training = False)
            results_dict["outputs"] = outputs
        return results_dict
    def training_step(self, features,nb_instances_in_global_batch):
        """
        Perform a training step on features and labels.

        Subclass and override to inject some custom behavior.
        """
        results_dict = self.run_model(features,prediction=False)
        scaled_loss = results_dict["loss"] / tf.cast(nb_instances_in_global_batch, dtype=results_dict["loss"].dtype)
        gradients = tf.gradients(scaled_loss, self.model.trainable_variables)
        gradients = [
            g if g is not None else tf.zeros_like(v) for g, v in zip(gradients, self.model.trainable_variables)
        ]

        if self.strategy.config.gradient_accumulation_steps > 1:
            self.gradient_accumulator(gradients)
        self.train_loss.update_state(scaled_loss)
        if self.strategy.config.gradient_accumulation_steps == 1:
            return gradients
    @tf.function
    def distributed_training_steps(self, batch):
        with self.strategy.strategy.scope():
            nb_instances_in_batch = self._compute_nb_instances(batch)
            inputs = self._get_step_inputs(batch, nb_instances_in_batch)
            self.strategy.strategy.run(self.apply_gradients, inputs)
    @staticmethod
    def _get_step_inputs(batch, nb_instances):
        featureNames = list(batch.keys())[-1]
        featureValue = batch[featureNames]

        if isinstance(featureValue, PerReplica):
            # need to make a `PerReplica` objects for ``nb_instances``
            nb_instances = PerReplica([nb_instances] * len(featureValue.values))

        step_inputs = (batch, nb_instances)

        return step_inputs
    @staticmethod
    def _compute_nb_instances(batch):
        featureNames = list(batch.keys())[-1]
        featureValues = batch[featureNames]
        if isinstance(featureValues, PerReplica):
            featureValues = tf.concat(featureValues.values, axis=0)
        nb_instances = featureValues.shape[0]

        return nb_instances
    def apply_gradients(self, features,nb_instances_in_global_batch):
        if self.strategy.config.gradient_accumulation_steps == 1:
            gradients = self.training_step(features,nb_instances_in_global_batch)
            
            gradients = [(tf.clip_by_value(grad, -self.strategy.config.max_grad_norm, self.strategy.config.max_grad_norm)) for grad in gradients
            ]
            self.optimizer.apply_gradients(list(zip(gradients, self.model.trainable_variables)))
        else:
            for _ in tf.range(self.strategy.config.gradient_accumulation_steps):
                reduced_features = {
                    k: ft[: self.strategy.config.train_batch_size] for k, ft in features.items()
                }
                
                self.training_step(reduced_features,nb_instances_in_global_batch)
                features = {
                    k: tf.concat(
                        [ft[self.strategy.config.train_batch_size:], reduced_features[k]],
                        axis=0,
                    )
                    for k, ft in features.items()
                }

            gradients = self.gradient_accumulator.gradients
            gradients = [(tf.clip_by_value(grad, -self.strategy.config.max_grad_norm, self.strategy.config.max_grad_norm)) for grad in gradients
            ]

            self.optimizer.apply_gradients(list(zip(gradients, self.model.trainable_variables)))
            self.gradient_accumulator.reset()
    def prepare(self):
        meta_file=os.path.join(self.strategy.config.processDir,
                               self.strategy.config.corpus,
                               self.strategy.config.lang,
                               "meta.json")
        #share MBERT vocab
        datasetPath = os.path.join(self.strategy.config.processDir,
                                   self.strategy.config.corpus,
                                   "databeam.pkl")
        datasetInstance = None
        if not os.path.exists(datasetPath):
            datasetInstance=DataSetBeam(self.strategy)
            with open(datasetPath,"wb") as f:
                pkl.dump(datasetInstance,f)
        else:
            with open(datasetPath,"rb") as f:
                datasetInstance = pkl.load(f)
        meta_dict=dict()
        if self.strategy.config.hier:
            num_train = datasetInstance.write_example_to_tfrecord(self.strategy.config.train_files,self.strategy.config.train_record%("hier"))
            num_dev = datasetInstance.write_example_to_tfrecord(self.strategy.config.dev_files,self.strategy.config.dev_record%("hier"))
            num_test = datasetInstance.write_example_to_tfrecord(self.strategy.config.test_files,self.strategy.config.test_record%("hier"))
        else:
            num_train = datasetInstance.write_example_to_tfrecord(self.strategy.config.train_files,self.strategy.config.train_record%("con"))
            num_dev = datasetInstance.write_example_to_tfrecord(self.strategy.config.dev_files,self.strategy.config.dev_record%("con"))
            num_test = datasetInstance.write_example_to_tfrecord(self.strategy.config.test_files,self.strategy.config.test_record%("con"))
        meta_dict["num_train"]=num_train
        meta_dict["num_dev"]=num_dev
        meta_dict["num_test"]=num_test
        self.meta_dict = meta_dict
        with open(meta_file,"w",encoding="utf-8") as f:
            json.dump(meta_dict, f)
    def train(self):
        def read_meta_files(lang):
            meta_file = os.path.join(self.strategy.config.processDir,
                                 self.strategy.config.corpus,
                                 lang,
                                 "meta.json")
            with open(meta_file,"r",encoding="utf-8") as f:
                meta_dict=json.load(f)
            return meta_dict
        
        
        def load_tfrecords_files(lang):
            if self.strategy.config.hier:
                train_record=os.path.join(self.strategy.config.processDir,
                                      self.strategy.config.corpus,
                                      lang,
                                      self.strategy.config.train_record%("hier"))
            else:
                train_record=os.path.join(self.strategy.config.processDir,
                                      self.strategy.config.corpus,
                                      lang,
                                      self.strategy.config.train_record%("con"))
            
            return train_record
            
            
        langs =["de","es","en","fr","it","ru","zh"]
        
        
        self.meta_dict = {}
        self.records = {}
        for lang in langs:
            self.meta_dict[lang] = read_meta_files(lang)
            self.records[lang] = load_tfrecords_files(lang)
        model_dir=os.path.join(self.strategy.config.model_dir,
                             self.strategy.config.corpus,
                             self.strategy.config.model)
        datasetPath = os.path.join(self.strategy.config.processDir,
                                   self.strategy.config.corpus,
                                   "databeam.pkl")
        with open(datasetPath,"rb") as f:
            datasetInstance = pkl.load(f)
        datasetInstance.update_strategy(self.strategy)
        tf.print("Info load data object...............")
        
        #the number of examples is equal
        num_update_steps_per_epoch = self.meta_dict["en"]["num_train"] / self.strategy.config.train_batch_size
        approx = math.floor if self.strategy.config.drop_last else math.ceil
        num_update_steps_per_epoch = approx(num_update_steps_per_epoch)
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        self.steps_per_epoch = num_update_steps_per_epoch
        total_batch_size = self.strategy.train_batch_size * self.strategy.config.gradient_accumulation_steps
        t_total = num_update_steps_per_epoch * self.strategy.config.num_train_epochs
        for lang in langs:
            train_dataset=datasetInstance.get_batchDataset(
                inputFile=self.records[lang],
                batch_size=total_batch_size,
                is_training=True)
            self.records[lang] = train_dataset
        tf.print("generate batch dataset...................")
        with self.strategy.strategy.scope():
            self.gradient_accumulator.reset()
            self.create_optimizer_and_scheduler(num_training_steps=t_total)
            if not self.model:
                embebPath = os.path.join(self.strategy.config.processDir,
                                         self.strategy.config.corpus,
                                         self.strategy.config.alignway,
                                         self.strategy.config.lang,
                                         "embed.pkl")
                embedhanel = open(embebPath,"rb")
                self.model = get_model(config = self.strategy.config,
                                       vocab_size = datasetInstance.vocabSize(),
                                       embedding_matrix=pkl.load(embedhanel), 
                                       PROM = datasetInstance.getSOSID(),#default, update prompt when generate
                                       EOS = datasetInstance.getEOSID(),
                                       PAD = datasetInstance.getPADID())
            iterations = self.optimizer.iterations
            checkpoint_prefix = os.path.join(model_dir, "ckpt")
            checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,model=self.model)
            #save all checkpoints for evaluate
            ckpt_manager = tf.train.CheckpointManager(checkpoint,checkpoint_prefix, max_to_keep=None)
            self.ckpt_handle = ckpt_manager
            steps_trained_in_current_epoch=0
            if ckpt_manager.latest_checkpoint and self.strategy.config.load_last_ckpt:
                tf.print("Checkpoint file %s found and restoring from checkpoint", ckpt_manager.latest_checkpoint)
                checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
                self.global_step =  iterations.numpy()
#                     epochs_trained = self.global_step // self.steps_per_epoch
                steps_trained_in_current_epoch = self.global_step % self.steps_per_epoch
            tf.summary.experimental.set_step(self.global_step)
            self.train_loss = tf.keras.metrics.Mean(name='train_loss')
            self.eval_loss = tf.keras.metrics.Mean(name="eval_loss")
            self.train_loss.reset_states()
            self.eval_loss.reset_states()
            start_time = datetime.datetime.now()
            multi_langs_train_dataset = [self.records[lang] for lang in langs]
            for features in zip(*multi_langs_train_dataset):
                """
                features: contain contains features of multiple languages
                """   
                features = combine_features(features)#
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                self.distributed_training_steps(features)
                self.global_step = iterations.numpy()
                if self.global_step % self.strategy.config.log_steps == 0:
                    tf.print("log info {} to step {} train loss {}".format((self.global_step-self.strategy.config.log_steps),self.global_step,self.train_loss.result()))
                    self.train_loss.reset_states()
                if self.global_step % self.strategy.config.eval_steps == 0:
                    #compute ppl
                    # metrics_result = self.evaluate(datasetInstance=datasetInstance)
                    # self.eval_loss.reset_states()
                    # formatString="the prediction metrics is "
                    # for _key in metrics_result:
                    #     formatString +=_key
                    #     formatString +=" {} ".format(metrics_result[_key])
                    # tf.print(formatString)
                    self.save_model()
            end_time = datetime.datetime.now()
            tf.print("Training took: {}".format(str(end_time - start_time)))
    # @tf.function
    def distributed_prediction_steps(self,batch,prediction):
        nb_instances_in_batch = self._compute_nb_instances(batch)
        inputs = self._get_step_inputs(batch, nb_instances_in_batch)
        inputs += (prediction,)
        logits = self.strategy.strategy.run(self.prediction_step, inputs)
        return logits
    def prediction_step(self,features,nb_instances_in_global_batch,prediction):
        results = self.run_model(features,prediction)
        if "loss" in results:
            scaled_loss = results["loss"] / tf.cast(nb_instances_in_global_batch, dtype=results["loss"].dtype)
            self.eval_loss.update_state(scaled_loss)
        return results
    def replicasDevice(self,replicas,container):
        def unstackList(numpyList):
            for item in numpyList:
                container.append(item)
        if self.strategy.n_replicas > 1:
            for val in replicas.values:
                unstackList(val.numpy())
        else:
            unstackList(replicas.numpy())
        return container
    def prediction_loop(self,dataset,prediction):
        preds_logits = list()
        real_labels = list()
        for _, features in enumerate(dataset):
            tgt = features["tgt"]
            results = self.distributed_prediction_steps(features,prediction) #batch * seq
            outputs = results["outputs"]
            real_labels = self.replicasDevice(tgt, real_labels)
            preds_logits = self.replicasDevice(outputs, preds_logits)
#             if "score" in results:
#                 final_scores = self.replicasDevice(results["score"], final_scores)
        
        return preds_logits, real_labels
    def calculationEmbedding(self,preddctions,references):
        """
        parameters:
            preddctions:["this is a demo","this is a demo"]
            references:["this is a demo","this is a demo"]
        """
        assert len(preddctions) == len(references)
        embed_path = os.path.join(self.strategy.config.EvalEmb,
                                  self.strategy.config.corpus,
                                  self.strategy.config.lang,
                                  "vectors.txt")
        dic ={}
        with open(embed_path,"r",encoding="utf-8") as f:
            for line in f:
                line  = line.rstrip()
                line = line.split()
                assert len(line) == self.strategy.config.emb_size +1
                dic[line[0]] = line[-self.strategy.config.emb_size:]
        
        preddctions = [prediction.split() for prediction in preddctions]
        references = [ref.split() for ref in references]
        ea_sum, vx_sum, gm_sum, counterp = 0, 0, 0, 0
        for rr, cc in tqdm(list(zip(references, preddctions))):
            ea_sum += cal_embedding_average(rr, cc, dic,self.strategy.config.emb_size)
            vx_sum += cal_vector_extrema(rr, cc, dic,self.strategy.config.emb_size)
            gm_sum += cal_greedy_matching_matrix(rr, cc, dic,self.strategy.config.emb_size)
            counterp += 1
        
        return ea_sum / counterp, gm_sum / counterp, vx_sum / counterp   
    def combineTokens(self,tokenList):
        for i in range(len(tokenList)-1,0,-1):
            if "##" in tokenList[i]:
                tokenList[i] = tokenList[i].replace("##","")
                tokenList[i-1] = tokenList[i-1] + tokenList[i]
                tokenList[i]=""
        tokenList=[token for token in tokenList if len(token)>0]
        if len(tokenList) > 0:
            if "##" in tokenList[0]:
                tokenList[0] = tokenList[0].replace("##","")
        return tokenList
    def removeDup(self,containerList):
        punc=[",",".","?"]
        i = 0
        while i < len(containerList)-1:
            if containerList[i] in punc and containerList[i+1] in punc:
                del containerList[i+1]
                i=0
                continue
            i = i+1
        return " ".join(containerList)
    def MtricsDecoder(self,dataobj,logits,outputFile,refFile="valid"):
        """
        parameters:
            logits:batch * seq
            label:batch * seq
        """
        predTokens,labelTokens,results=[],[],[]
        refFileName = os.path.join(self.strategy.config.processDir,
                                   self.strategy.config.corpus,
                                   self.strategy.config.alignway,
                                   self.strategy.config.lang,
                                   refFile+".ref")
        #filter special ID <sos> <eos> <pad> user0 user1
        eos_id = dataobj.getEOSID()
        pad_id = dataobj.getPADID()
        unk_id = dataobj.getUNKID()
        for seqIds in logits:
            seqIds = seqIds.tolist()
            if eos_id in seqIds:
                _index = seqIds.index(eos_id)
                seqIds = seqIds[:_index]
            if pad_id in seqIds:
                _index = seqIds.index(pad_id)
                seqIds = seqIds[:_index]
            while unk_id in seqIds:
                seqIds.remove(unk_id)
            tokens = " ".join(dataobj.convert_id2text(seqIds))
            tokens = tokens.replace(dataobj.SOS,"")
            if self.strategy.config.do_lower_case:
                predTokens.append(tokens.lower())
            else:
                predTokens.append(tokens)
        with open(refFileName,"r",encoding="utf-8") as f:
            for line in f:
                tokens = line.rstrip()
                if self.strategy.config.do_lower_case:
                    labelTokens.append(tokens.lower())
                else:
                    labelTokens.append(tokens)
        #embedding metrics
        ea_sum,gm_sum,vx_sum= self.calculationEmbedding(predTokens,labelTokens)
        predTokens = [self.combineTokens(tokens.split()) for tokens in predTokens] 
        predTokens = [self.removeDup(tokens) for tokens in predTokens]
        labelTokens = [self.combineTokens(tokens.split()) for tokens in labelTokens]
        labelTokens = [self.removeDup(tokens) for tokens in labelTokens]
        distinct_result = cal_Distinct(predTokens)
        predDict = {i:[pred] for i,pred in enumerate(predTokens)}
        labelDict = {i:[ref] for i,ref in enumerate(labelTokens)}
        nltkBleu = cal_corpus_bleu(labelTokens,predTokens)
        metricsResult = compute_bleu_rouge_single_prediction(predDict,labelDict)
        metricsResult["EA"] = ea_sum
        metricsResult["GA"] = gm_sum
        metricsResult["VX"] = vx_sum
        for _key,_value in distinct_result.items():
            metricsResult[_key] = _value
        for _key,_value in nltkBleu.items():
            metricsResult[_key] = _value
        for keyId,tokens in predDict.items():
            results.append({"pred":tokens[0],
                            "tgt":labelDict[keyId][0]})
        with open(outputFile,"w",encoding="utf-8") as fp:
            json.dump(results,fp,ensure_ascii=False,indent=4)
        return metricsResult
    def predict(self,datasetInstance = None):
        if datasetInstance is None:
            datasetPath = os.path.join(self.strategy.config.processDir,self.strategy.config.corpus,"databeam.pkl")
            with open(datasetPath,"rb") as f:
                datasetInstance = pkl.load(f)
        if self.strategy.config.hier:
            test_record=os.path.join(self.strategy.config.processDir,self.strategy.config.corpus,self.strategy.config.test_record%("hier"))
        else:
            test_record=os.path.join(self.strategy.config.processDir,self.strategy.config.corpus,self.strategy.config.test_record%("con"))
        test_dataset = datasetInstance.get_batchDataset(
            inputFile=test_record,
            batch_size=self.strategy.eval_batch_size,
            is_training=False)
        embebPath = os.path.join(self.strategy.config.processDir,
                                         self.strategy.config.corpus,
                                         "embed.pkl")
        embedhanel = open(embebPath,"rb")
        self.model = get_model(config = self.strategy.config,
                                vocab_size = datasetInstance.vocabSize(),
                                embedding_matrix=pkl.load(embedhanel),
                                SOS = datasetInstance.getPROM(self.strategy.config.lang),
                                EOS = datasetInstance.getEOSID(),
                                PAD = datasetInstance.getPADID())
        model_dir=os.path.join(self.strategy.config.model_dir,
                                  self.strategy.config.corpus,
                                  self.strategy.config.lang,
                                  self.strategy.config.model)
        
        checkpoint_path = os.path.join(model_dir, "model.ckpt-best")
        self.model.load_weights(checkpoint_path)
        preds_logits, real_labels= self.prediction_loop(test_dataset,prediction=True)
        
        outputFile = os.path.join(self.strategy.config.model_dir,
                                  self.strategy.config.corpus,
                                  self.strategy.config.alignway,
                                  self.strategy.config.lang,
                                  self.strategy.config.model,"predictions.json")
        metricsResult = self.MtricsDecoder(dataobj=datasetInstance, 
                               logits=preds_logits,
                               outputFile=outputFile,
                               refFile="test")
        formatString="the prediction metrics is "
        for _key in metricsResult:
            formatString +=_key
            formatString +=" {} ".format(metricsResult[_key])
        tf.print(formatString)
    def _find_valid_cands(self,model_dir):
        filenames = tf.io.gfile.listdir(os.path.join(model_dir, "ckpt"))
        steps_and_files = {}
        for filename in filenames:
            if filename.endswith(".index"):
                ckpt_name = filename[:-6]
                cur_filename = os.path.join(os.path.join(model_dir, "ckpt"), ckpt_name)
                gstep = int(cur_filename.split("-")[-1])
                if gstep not in steps_and_files:
                    tf.print("Add {} to eval list.".format(cur_filename))
                    steps_and_files[gstep] = cur_filename
        return steps_and_files
    def evaluate(self,datasetInstance = None,only_evaluation=False):
        if datasetInstance is None:
            datasetPath = os.path.join(self.strategy.config.processDir,
                                       self.strategy.config.corpus,
                                       "databeam.pkl")
            with open(datasetPath,"rb") as f:
                datasetInstance = pkl.load(f)
        if self.strategy.config.hier:
            dev_record=os.path.join(self.strategy.config.processDir,
                                    self.strategy.config.corpus,
                                    self.strategy.config.lang,
                                    self.strategy.config.dev_record%("hier"))
        else:
            dev_record=os.path.join(self.strategy.config.processDir,
                                    self.strategy.config.corpus,
                                    self.strategy.config.lang,
                                    self.strategy.config.dev_record%("con"))
        dev_dataset = datasetInstance.get_batchDataset(
            inputFile=dev_record,
            batch_size=self.strategy.eval_batch_size,
            is_training=False)
        outputFile = os.path.join(self.strategy.config.model_dir,
                                  self.strategy.config.corpus,
                                  self.strategy.config.lang,
                                  self.strategy.config.model,"eval.json")
        outputModel = os.path.join(self.strategy.config.model_dir,
                                  self.strategy.config.corpus,
                                  self.strategy.config.lang,
                                  self.strategy.config.model)
        writer = tf.io.gfile.GFile(outputModel, "w")
        model_dir=os.path.join(self.strategy.config.model_dir,
                             self.strategy.config.corpus,
                             self.strategy.config.model)
        best_em = -1
        if only_evaluation:
            embebPath = os.path.join(self.strategy.config.processDir,
                                         self.strategy.config.corpus,
                                         "embed.pkl")
            embedhanel = open(embebPath,"rb")
            self.model = get_model(config = self.strategy.config,
                                    vocab_size = datasetInstance.vocabSize(),
                                    embedding_matrix=pkl.load(embedhanel),
                                    PROM = datasetInstance.getPROM(self.strategy.config.lang),
                                    EOS = datasetInstance.getEOSID(),
                                    PAD = datasetInstance.getPADID())
            model_dir=os.path.join(self.strategy.config.model_dir,
                                   self.strategy.config.corpus,
                                   self.strategy.config.model)
            steps_and_files = self._find_valid_cands(model_dir)
            
            for ele in sorted(steps_and_files.items()):
                step, checkpoint_path = ele
                self.model.load_weights(checkpoint_path)
                tf.print("Checkpoint file found and restoring from checkpoint",checkpoint_path)
                preds_logits, real_labels = self.prediction_loop(dev_dataset,prediction=True)
                metricsResult = self.MtricsDecoder(dataobj=datasetInstance, 
                                   logits=preds_logits,
                                   outputFile=outputFile,
                                   refFile="test")
                formatString="the prediction metrics is "
                for _key in metricsResult:
                    formatString +=_key
                    formatString +=" {} ".format(metricsResult[_key])
                tf.print(formatString)
                if metricsResult["EA"] > best_em:
                    best_em = metricsResult["EA"]
                    for ext in ["meta", "data-00000-of-00001", "index"]:
                        src_ckpt = checkpoint_path + ".{}".format(ext)
                        tgt_ckpt = checkpoint_path.rsplit("-", 1)[0] + "-best.{}".format(ext)
                        tf.io.gfile.copy(src_ckpt, tgt_ckpt, overwrite=True)
                        writer.write("saved {} to {}\n".format(src_ckpt, tgt_ckpt))
                
        preds_logits, real_labels = self.prediction_loop(dev_dataset,prediction=True)
        metricsResult = self.MtricsDecoder(dataobj=datasetInstance, 
                                   logits=preds_logits, 
                                   outputFile=outputFile,
                                   refFile="test")
        ppl = math.exp(float(self.eval_loss.result()))
        metricsResult["ppl"] = ppl
        return metricsResult
    def save_model(self):
        save_path = self.ckpt_handle.save()
        tf.print("save the model {}".format(save_path))
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        TFTrainer's init through :obj:`optimizers`, or subclass and override this method.
        """
        if not self.optimizer and not self.lr_scheduler:
            self.optimizer, self.lr_scheduler = create_optimizer(
                self.strategy.config.learning_rate,
                num_training_steps,
                self.strategy.config.warmup_steps,
                weight_decay_rate=self.strategy.config.weight_decay,
                adam_beta1=self.strategy.config.adam_beta1,
                adam_beta2=self.strategy.config.adam_beta2,
                adam_epsilon=self.strategy.config.adam_epsilon,
                
            )
    














