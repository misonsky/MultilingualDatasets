#coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow import keras
from utils.beam_search import beam_search
loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
class BASICEncoder(keras.Model):
    def __init__(self, rnn_type,output_size,num_layers=1,bidirectional=False):
        super(BASICEncoder, self).__init__()
        assert rnn_type in ['GRU','gru','LSTM','lstm']
        if bidirectional:
            assert output_size % 2 == 0
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        units = int(output_size / self.num_directions)
        if rnn_type == 'GRU' or rnn_type == 'gru':
            rnnCell = [getattr(keras.layers, 'GRUCell')(units) for _ in range(num_layers)]
        else:
            rnnCell = [getattr(keras.layers, 'LSTMCell')(units) for _ in range(num_layers)]
        self.rnn = keras.layers.RNN(rnnCell,
                                    return_sequences=True, 
                                    return_state=True)
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        if bidirectional:
            self.rnn = keras.layers.Bidirectional(self.rnn)
        self.bidirectional = bidirectional
    def call(self, x, mask,initial_state=None):  # [batch, timesteps, input_dim]
        outputs=  self.rnn(x,
                           mask=mask,
                           initial_state = initial_state)
        output = outputs[0] #batch * seq * d
        states = outputs[1:] #(num *bidirec) * batch * d
        return output,states
class Encoder(keras.Model):
    def __init__(self,embedFunction,output_size,config,bidirectional=False):
        super(Encoder, self).__init__()
        self.embedding = embedFunction
        self.encoder = BASICEncoder(rnn_type=config.rnn_type,
                                    output_size=output_size,
                                    num_layers=config.num_layers,
                                    bidirectional=bidirectional)
    def call(self, x, mask,hidden=None,useEmbedding=True):
        if useEmbedding:
            x = self.embedding(x)
        output, state = self.encoder(x,mask=mask,initial_state = hidden)
        return output, state
class BahdanauAttention(keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    def call(self, query, values):
        """
        parameter:
            query:(batch * d)
            values:(batch * seq * d)
        return:
            context_vector:(batch *d)
        """
        #(batch * d)----->(batch * 1 * d)
        hidden_with_time_axis = tf.expand_dims(query, 1)#(batch * 1 * d)
        #score:(batch * seq * 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        # context_vector:(batch * seq * d)
        context_vector = attention_weights * values
        # context_vector:(batch * d )
        context_vector = tf.reduce_sum(context_vector, axis=1)
    
        return context_vector
class Decoder(keras.Model):
    def __init__(self, config,embedFunction,bidirectional=False):
        super(Decoder, self).__init__()
        self.embedding = embedFunction
        self.decoder = BASICEncoder(rnn_type=config.rnn_type,
                                    output_size=config.d_model,
                                    num_layers=config.num_layers,
                                    bidirectional=bidirectional)
        self.attention = BahdanauAttention(config.d_model)
    def call(self, x, hidden,enc_output):
        """
        parameter:
            enc_output: output of encoder (batch * seq * d)
            x:input of decoder (batch *1 * hidden)
            hidden:previous state (batch * d)
        """
        x = self.embedding(x)
        context_vector = self.attention(hidden[-1], enc_output)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.decoder(x,mask=None,initial_state=hidden)
        output = tf.reshape(output, (-1, output.shape[2]))#(batch * 1) * hidden
    
        return output, state

class FFN(keras.layers.Layer):
    def __init__(self,output_size):
        super(FFN, self).__init__()
        self.dense1 = keras.layers.Dense(output_size*2)
        self.dense2 = keras.layers.Dense(output_size)
    def call(self,x,activation=None):
        x = self.dense1(x)
        if activation is not None:
            x = activation(x)
        x = self.dense2(x)
        if activation is not None:
            x = activation(x)
        return x

class VariableLayer(keras.Model):
    def __init__(self, context_hidden, encoder_hidden, z_hidden,config):
        super(VariableLayer, self).__init__()
        self.config = config
        self.dytype = tf.float16 if config.fp16 else tf.float32
        self.context_hidden = context_hidden
        self.encoder_hidden = encoder_hidden
        self.z_hidden = z_hidden
        self.sent_prior_h = keras.layers.Dense(config.d_model)
        self.sent_prior_mu = keras.layers.Dense(config.d_model)
        self.sent_prior_var = keras.layers.Dense(config.d_model)
        
        self.sent_posterior_h = FFN(config.d_model)
        self.sent_posterior_mu = keras.layers.Dense(config.d_model)
        self.sent_posterior_var = keras.layers.Dense(config.d_model)
        self.conv_posterior_h = FFN(config.d_model)
        self.conv_posterior_mu = keras.layers.Dense(config.d_model)
        self.conv_posterior_var = keras.layers.Dense(config.d_model)
        self.noraml_initializer = keras.initializers.random_normal(mean=0., stddev=1.)
    def conv_prior(self):
        # Standard gaussian prior
        return tf.constant([0.0],dtype=self.dytype),tf.constant([1.0],dtype=self.dytype)
    def prior(self, context_outputs):
        # context_outputs: [batch, context_hidden]
        h_prior = self.prior_h(context_outputs,activation = tf.nn.tanh)
        mu_prior = self.prior_mu(h_prior)
        var_prior = tf.nn.softplus(self.prior_var(h_prior))
        return mu_prior, var_prior
    def conv_posterior(self, context_inference_hidden):
        h_posterior = self.conv_posterior_h(context_inference_hidden)
        mu_posterior = self.conv_posterior_mu(h_posterior)
        var_posterior = tf.nn.softplus(self.conv_posterior_var(h_posterior))
        return mu_posterior, var_posterior
    def sent_prior(self, context_outputs, z_conv):
        # Context dependent prior
        h_prior = self.sent_prior_h(tf.concat([context_outputs, z_conv],axis=1))
        mu_prior = self.sent_prior_mu(h_prior)
        var_prior = tf.nn.softplus(self.sent_prior_var(h_prior))
        return mu_prior, var_prior
    def sent_posterior(self, context_outputs, encoder_hidden, z_conv):
        h_posterior = self.sent_posterior_h(tf.concat([context_outputs, encoder_hidden, z_conv],axis=1))
        mu_posterior = self.sent_posterior_mu(h_posterior)
        var_posterior = tf.nn.softplus(self.sent_posterior_var(h_posterior))
        return mu_posterior, var_posterior
    def normal_logpdf(self,x, mean, var):
        """
        Args:
            x: (Variable, FloatTensor) [batch_size, dim]
            mean: (Variable, FloatTensor) [batch_size, dim] or [batch_size] or [1]
            var: (Variable, FloatTensor) [batch_size, dim]: positive value
        Return:
            log_p: (Variable, FloatTensor) [batch_size]
        """
        log2pi = tf.math.log(2. * np.pi)
        return 0.5 * tf.reduce_sum(-log2pi -tf.math.log(var) - (tf.pow(x-mean,2.0) /var),axis=1)
    def kl_div(self, mu1, var1, mu2, var2):
        one = tf.constant([1.0])
        kl_div = tf.reduce_sum(0.5 * (tf.math.log(var2)-tf.math.log(var1) + (var1 + tf.pow(mu1 - mu2,2.0)) / var2 -one),axis=1)
        return kl_div
    def vconv_context(self,context_inference_hidden=None,training=True):
        conv_mu_prior, conv_var_prior = self.conv_prior()
        conv_eps = self.noraml_initializer(shape=(context_inference_hidden.shape[0], context_inference_hidden.shape[-1]))
        if training:
            conv_mu_posterior, conv_var_posterior = self.conv_posterior(context_inference_hidden)
            z_conv = conv_mu_posterior + tf.math.sqrt(conv_var_posterior) * conv_eps
            log_q_zx_conv = tf.reduce_sum(self.normal_logpdf(z_conv, conv_mu_posterior, conv_var_posterior))
            log_p_z_conv = tf.reduce_sum(self.normal_logpdf(z_conv, conv_mu_prior, conv_var_prior))
            kl_div_conv = tf.reduce_sum(self.kl_div(conv_mu_posterior, conv_var_posterior,conv_mu_prior, conv_var_prior))
        else:
            z_conv = tf.math.sqrt(conv_var_prior) * conv_eps
            log_q_zx_conv = None
            log_p_z_conv = None
            kl_div_conv = None
        return z_conv,conv_mu_prior,conv_var_prior,log_q_zx_conv,log_p_z_conv,kl_div_conv
    def vconv_sent(self,context_outputs,z_conv,conv_mu_prior,conv_var_prior,kl_div_conv,log_q_zx_conv,log_p_z_conv,encoder_flat=None,training=False):
        sent_mu_prior, sent_var_prior = self.sent_prior(context_outputs, z_conv)
        eps = self.noraml_initializer(shape=(context_outputs.shape[0], context_outputs.shape[-1]))
        if training:
            sent_mu_posterior, sent_var_posterior = self.sent_posterior(context_outputs, encoder_flat, z_conv)
            z_sent = sent_mu_posterior + tf.math.sqrt(sent_var_posterior) * eps
            log_q_zx_sent = tf.reduce_sum(self.normal_logpdf(z_sent, sent_mu_posterior, sent_var_posterior))
            log_p_z_sent = tf.reduce_sum(self.normal_logpdf(z_sent, sent_mu_prior, sent_var_prior))
            kl_div_sent = tf.reduce_sum(self.kl_div(sent_mu_posterior, sent_var_posterior, sent_mu_prior, sent_var_prior))
            kl_div = kl_div_conv + kl_div_sent
            log_q_zx = log_q_zx_conv + log_q_zx_sent
            log_p_z = log_p_z_conv + log_p_z_sent
        else:
            z_sent = sent_mu_prior + tf.math.sqrt(sent_var_prior) * eps
            kl_div = None
            log_p_z = tf.reduce_sum(self.normal_logpdf(z_sent, sent_mu_prior, sent_var_prior))
            log_p_z += tf.reduce_sum(self.normal_logpdf(z_conv, conv_mu_prior, conv_var_prior))
            log_q_zx = None
        return z_sent,kl_div, log_p_z, log_q_zx
class VHCR(keras.Model):
    def __init__(self,vocab_size,embedding_dim,matrix,config,PROM=0,EOS=0,PAD=0):
        super(VHCR, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                   embedding_dim,
                                                   embeddings_initializer=keras.initializers.constant(matrix),
                                                   trainable=True)
        self.encoder1 = Encoder(embedFunction=self.embedding,
                                output_size = config.d_model,
                                config=config,
                                bidirectional=config.bidirectional)
        self.encoder2 = Encoder(embedFunction=self.embedding,
                                output_size = config.d_model,
                                config=config,
                                bidirectional=config.bidirectional)
        self.encoder3 = Encoder(embedFunction=self.embedding,
                                output_size = config.d_model,
                                config=config,
                                bidirectional = config.bidirectional)#inference encoder
        self.decoder = Decoder(config=config,embedFunction=self.embedding,bidirectional=False)
        self.variablelayer = VariableLayer(config.d_model, config.d_model, config.d_model,config)
        self.z_conv2context = keras.layers.Dense(config.d_model)
        self.dense1 = keras.layers.Dense(config.d_model)
        self.teach_force = config.teach_force
        self.config = config
        self.output_size = vocab_size
        self.SOS = PROM
        self.EOS = EOS
        self.PAD = PAD
        self.kl_mult = 0.0
    def getEmbeddingTable(self):
        if self.config.fp16:
            return tf.cast(self.embedding.embeddings,dtype=tf.float16)
        return self.embedding.embeddings
    def outputLayer(self,logits):
        """
        parameters:
            logits:batch * d
        return: batch  * vocabSize
        """
        return tf.einsum("bd,vd->bv",logits,self.getEmbeddingTable())
        
    def masked_fill(self,t, mask, value=-float('inf')):
        return t * (1 - tf.cast(mask, tf.float32)) + value * tf.cast(mask, tf.float32)
    def loss_function(self,real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, self.PAD))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)
    def stackStates(self,states,bidirectional):
        concat_states =[]#num_layer * batch * d
        if bidirectional:
            for fw,bw in zip(states[:self.config.num_layers],states[self.config.num_layers:]):
                concat_states.append(tf.concat([fw,bw],axis=-1))
        else:
            concat_states = states
        return concat_states
    @tf.function
    def call(self,features,training=True):
        """
        parameters:
            src: batch * (max_turn * max_seq)
            tgt: batch * max_seq 
        """
        src = features["src"]
        tgt = features["tgt"]
        src = tf.reshape(src,shape=[src.shape[0],self.config.max_turn,self.config.max_utterance_len])
        src = tf.unstack(src,axis=1)
        utterances,context_mask= [],[]
        for utt in src:
            mask = tf.not_equal(utt,self.PAD) #batch * seq
            mask = tf.cast(mask,dtype=tf.zeros(1).dtype)
            context_mask.append(tf.reduce_sum(mask,axis=1)) 
            _,state = self.encoder1(utt,mask=mask)#(num * bid) * batch * d
            state = self.stackStates(state, bidirectional=True)
            state = tf.transpose(tf.stack(state,axis=0),perm=[1,0,2])
            utterances.append(tf.reduce_sum(state,axis=1))
        encoder_context = tf.stack(utterances,axis=1)#batch * max_turn * hidden
        encoder_context_mask = tf.cast(tf.stack(context_mask,axis=1),dtype=tf.zeros(1).dtype) #batch * max_turn
        _,inference_hidden = self.encoder2(tgt,mask=tf.cast(tf.not_equal(tgt,self.PAD),dtype=tf.zeros(1).dtype))#(num * bid) * batch * d
        inference_hidden = tf.stack(self.stackStates(inference_hidden,bidirectional=True),axis=0)#num_layers * batch * hidden]
        inference_hidden = tf.reduce_sum(inference_hidden,axis=0)
        z_conv,conv_mu_prior,conv_var_prior,log_q_zx_conv,log_p_z_conv,kl_div_conv = self.variablelayer.vconv_context(inference_hidden,training=True)
        context_init = self.z_conv2context(z_conv)
        if self.config.bidirectional:
            context_init = tf.reshape(context_init, shape=[2,context_init.shape[0],context_init.shape[1] // 2])
            context_init = tf.tile(context_init,multiples=[self.config.num_layers,1,1])
        else:
            context_init = tf.tile(tf.expand_dims(z_conv,axis=0),multiples=[self.config.num_layers,1,1])
        context_init = tf.unstack(context_init)
        z_conv_expand = tf.tile(tf.expand_dims(z_conv,axis=1),multiples=[1,encoder_context.shape[1],1])
        context_outputs, context_hidden= self.encoder3(tf.concat([encoder_context,z_conv_expand],axis=2),
                                                        mask=encoder_context_mask,
                                                        hidden=context_init,
                                                        useEmbedding=False)
        flat_context_hidden = tf.reduce_sum(tf.stack(self.stackStates(context_hidden,bidirectional=True),axis=0),axis=0)#num_layers * batch * hidden]
#         context_outputs_flat = tf.reshape(context_outputs,shape=[context_outputs.shape[0]*context_outputs.shape[1],context_outputs.shape[-1]])
#         z_conv_flat = tf.reshape(z_conv_expand,shape=[z_conv_expand.shape[0]*z_conv_expand.shape[1],z_conv_expand.shape[-1]])
        z_sent,kl_div, log_p_z, log_q_zx = self.variablelayer.vconv_sent(context_outputs=flat_context_hidden,
                                      z_conv = z_conv,
                                      conv_mu_prior = conv_mu_prior,
                                      conv_var_prior = conv_var_prior, 
                                      kl_div_conv = kl_div_conv, 
                                      log_q_zx_conv = log_q_zx_conv, 
                                      log_p_z_conv = log_p_z_conv, 
                                      encoder_flat = inference_hidden,
                                      training=True)
        context_hidden = self.stackStates(context_hidden, bidirectional=True)
        z_sent = tf.tile(tf.expand_dims(z_sent,axis=0),multiples=[len(context_hidden),1,1])
        z_conv = tf.tile(tf.expand_dims(z_conv,axis=0),multiples=[len(context_hidden),1,1])
        latent_context = tf.concat([context_hidden, z_sent, z_conv],axis=-1)
        latent_context = self.dense1(latent_context)
        latent_context = tf.unstack(latent_context,axis=0)#list[num,batch,d]
        loss = 0
        dec_input = tf.expand_dims(tgt[:,0],1)# batch * 1
        for t in range(1, tgt.shape[1]):
            decoderOut, dec_hidden = self.decoder(x = dec_input, 
                         hidden = latent_context,
                         enc_output = context_outputs)
            logits = self.outputLayer(decoderOut)#batch * vocab
            dec_hidden = dec_hidden
            loss += self.loss_function(tgt[:, t], logits)
            dec_input = tf.expand_dims(tgt[:,t], 1)
        loss = loss / tf.reduce_sum(tf.cast(tf.not_equal(tgt[:,1:],self.PAD),dtype=loss.dtype))
        loss += self.kl_mult * kl_div
        self.kl_mult = min(self.kl_mult + 1.0 / self.config.kl_annealing_iter, 1.0)
        return _,loss
    def BeamDecoder(self,features):
        src = features["src"]
        batchSize = src.shape[0]
        src = tf.reshape(src,shape=[src.shape[0],self.config.max_turn,self.config.max_utterance_len])
        src = tf.unstack(src,axis=1)
        utterances,context_mask= [],[]
        for utt in src:
            mask = tf.not_equal(utt,self.PAD)
            mask = tf.cast(mask,dtype=tf.zeros(1).dtype)
            context_mask.append(tf.reduce_sum(mask,axis=1))
            _,state = self.encoder1(utt,mask=mask)
            state = self.stackStates(state, bidirectional=True)
            state = tf.transpose(tf.stack(state,axis=0),perm=[1,0,2])
            utterances.append(tf.reduce_sum(state,axis=1))
        encoder_context_mask = tf.cast(tf.cast(tf.stack(context_mask,axis=1),dtype=tf.bool),dtype=tf.zeros(1).dtype)
        encoder_context = tf.stack(utterances, axis=1)#batch * max_turn * hidden
        z_conv,conv_mu_prior,conv_var_prior,log_q_zx_conv,log_p_z_conv,kl_div_conv = self.variablelayer.vconv_context(encoder_context,training=False)
        z_conv_expand = tf.tile(tf.expand_dims(z_conv,axis=1),multiples=[1,encoder_context.shape[1],1])
        context_init = self.z_conv2context(z_conv)
        if self.config.bidirectional:
            context_init = tf.reshape(context_init, shape=[2,context_init.shape[0],context_init.shape[1] // 2])
            context_init = tf.tile(context_init,multiples=[self.config.num_layers,1,1])
        else:
            context_init = tf.tile(tf.expand_dims(z_conv,axis=0),multiples=[self.config.num_layers,1,1])
        context_init = tf.unstack(context_init)
        context_outputs, context_hidden = self.encoder3(tf.concat([encoder_context,z_conv_expand],axis=2),
                                      mask=encoder_context_mask,
                                      hidden=context_init,
                                      useEmbedding=False)
        
        flat_context_hidden = tf.reduce_sum(tf.stack(self.stackStates(context_hidden,bidirectional=True),axis=0),axis=0)#num_layers * batch * hidden]
        z_sent,kl_div, log_p_z, log_q_zx = self.variablelayer.vconv_sent(context_outputs=flat_context_hidden,
                                      z_conv = z_conv,
                                      conv_mu_prior = conv_mu_prior,
                                      conv_var_prior = conv_var_prior, 
                                      kl_div_conv = kl_div_conv, 
                                      log_q_zx_conv = log_q_zx_conv, 
                                      log_p_z_conv = log_p_z_conv, 
                                      encoder_flat=flat_context_hidden,
                                      training=False)
        context_hidden = self.stackStates(context_hidden, bidirectional=True)
#         print(context_hidden.shape)
        z_sent = tf.tile(tf.expand_dims(z_sent,axis=0),multiples=[len(context_hidden),1,1])
        z_conv = tf.tile(tf.expand_dims(z_conv,axis=0),multiples=[len(context_hidden),1,1])
        latent_context = tf.concat([context_hidden, z_sent, z_conv],axis=-1)
        latent_context = self.dense1(latent_context)
        latent_context = tf.unstack(latent_context,axis=0)#list[num,batch,d]
        startIdx = [self.SOS] * batchSize
        states = {"dec_hidden":latent_context,"context":context_outputs}
        def symbols_to_logits_fn(tgtids, i, states):
            """
                tgtids:batch * seq
            """
            dec_input = tf.expand_dims(tgtids[:,i],axis=1)# batch * 1
            decoderOut, dec_hidden = self.decoder(x = dec_input, 
                        hidden = states["dec_hidden"],
                        enc_output = states["context"])
            logits = self.outputLayer(decoderOut)#batch * vocab
            states["dec_hidden"] = dec_hidden
            return logits, states
        ids, scores=beam_search(symbols_to_logits_fn=symbols_to_logits_fn,
                initial_ids=startIdx,
                beam_size=self.config.beam_size,
                decode_length=self.config.decode_length,
                vocab_size=self.output_size,
                alpha=self.config.alpha,
                states=states,
                eos_id=self.EOS,
                stop_early=True)
        
        return ids[:,0,:]
        
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

