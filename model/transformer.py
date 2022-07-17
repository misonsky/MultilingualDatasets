#coding=utf-8
import tensorflow as tf
from tensorflow import keras
from utils.beam_search import beam_search
from utils.transformer_utils import point_wise_feed_forward_network,scaled_dot_product_attention,positional_encoding
from utils.transformer_utils import create_look_ahead_mask,create_padding_mask,create_masks
loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                         reduction=tf.keras.losses.Reduction.NONE)


class Normalization(keras.layers.Layer):
    def __init__(self,config):
        super(MultiHeadAttention, self).__init__()
        self.dytype = tf.float16 if config.fp16 else tf.float32
    def build(self, input_shape):
        self.beta = self.add_weight("beta", 
                                    shape=input_shape[-1:], 
                                    dtype = self.dytype, 
                                    initializer= keras.initializers.Zeros(), 
                                    trainable=True)
        self.gamma = self.add_weight("gamma", 
                                    shape=input_shape[-1:], 
                                    dtype = self.dytype, 
                                    initializer= keras.initializers.Zeros(), 
                                    trainable=True)
        
    def call(self,inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        normalized = (inputs - mean) / ((variance + self.epsilon) ** (.5))
        outputs = self.gamma * normalized + self.beta
        return outputs
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
    def split_heads(self, x, batch_size):
        """
            return (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    def call(self,query,memory,mask,cache=None):
        batch_size = query.shape[0]
        if memory is None:
            q = self.wq(query) # (batch_size, seq_len, d_model)
            k = self.wk(query) # (batch_size, seq_len, d_model)
            v = self.wv(query) # (batch_size, seq_len, d_model)
            
            k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
            v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
            if cache is not None:
                k = cache["k"] = tf.concat([cache["k"], k], axis=2)
                v = cache["v"] = tf.concat([cache["v"], v], axis=2)
        else:
            # encoder-decoder attention
            q = self.wq(query)
            if cache is not None:
                k = cache["k_encdec"]
                v = cache["v_encdec"]
            else:
                k = self.wk(memory) # (batch_size, seq_len, d_model)
                v = self.wv(memory) # (batch_size, seq_len, d_model)
                k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
                v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        q *= q.shape[0] ** -0.5
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    def call(self, x, training, mask):
        attn_output = self.mha(query=x, 
                               memory=None,
                               mask = mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
    
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
    
        self.ffn = point_wise_feed_forward_network(d_model, dff)
     
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
    def GetSelfAtt(self):
        return self.mha1
    def GetCrossfAtt(self):
        return self.mha2
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask,cache=None):
        """
        parameter:
                x:decoder input
                enc_output:(batch_size, input_seq_len, d_model)
        """
        attn1 = self.mha1(query=x, 
                        memory=None,
                        mask = look_ahead_mask,
                        cache = cache) # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        attn2 = self.mha2(query = out1,
                          memory = enc_output,
                          mask = padding_mask,
                          cache = cache)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
        
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
        
        return out3

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
      
        self.dropout = tf.keras.layers.Dropout(rate)
    def call(self, x, training, mask):

        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x  # (batch_size, input_seq_len, d_model)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,rate=0.1):
        super(Decoder, self).__init__()
    
        self.d_model = d_model
        self.num_layers = num_layers
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(rate)
    def getDecoderLayer(self,i):
        return self.dec_layers[i]
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask,cache=None):
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            if cache is not None:
                x = self.dec_layers[i](x, enc_output, training,look_ahead_mask, padding_mask,cache=cache[str(i)])
            else:
                x = self.dec_layers[i](x, enc_output, training,look_ahead_mask, padding_mask,cache=None)
        # x.shape == (batch_size, target_seq_len, d_model)
        return x

class HierTransformer(tf.keras.Model):
    def __init__(self,vocab_size,matrix,config,PROM=0,EOS=0,PAD=0,rate=0.1):
        super(HierTransformer, self).__init__()
        self.SOS = PROM
        self.EOS = EOS
        self.PAD = PAD
        self.config = config
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                   config.emb_size,
                                                   embeddings_initializer=keras.initializers.constant(matrix),
                                                   trainable=True)
        self.tokens_encoding = positional_encoding(config.max_utterance_len, config.emb_size)
        self.turns_encoding = positional_encoding(config.max_turn, config.emb_size)
        self.encoder1 = Encoder(num_layers=config.num_layers, 
                               d_model=config.d_model, 
                               num_heads=config.num_heads, 
                               dff=config.dff,
                               rate=rate)
        
        self.encoder2 = Encoder(num_layers=config.num_layers, 
                               d_model=config.d_model, 
                               num_heads=config.num_heads, 
                               dff=config.dff,
                               rate=rate)
    
        self.decoder = Decoder(num_layers=config.decoder_layers, 
                               d_model=config.d_model,
                               num_heads=config.num_heads,
                               dff=config.dff,
                               rate=rate)
#         self.dense1 = keras.layers.Dense(1,activation="gelu")
        self.output_size = vocab_size
    def convert2embed(self,x,positions):
        seq_len = x.shape[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.config.d_model, tf.float32))
        x += positions[:, :seq_len, :]
        return x
        
    def getEmbeddingTable(self):
        return self.embedding.embeddings
    def outputLayer(self,logits):
        """
        parameters:
            logits:batch * seq * d
        return: batch  * seq * vocabSize
        """
        return tf.einsum("bsd,vd->bsv",logits,self.getEmbeddingTable())
    def loss_function(self,real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, self.PAD))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    @tf.function
    def call(self, features, training):
        inp = features["src"]
        tar = features["tgt"]
        enc_padding_mask, look_ahead_mask, _ = create_masks(inp,tar[:,:-1],padId=self.PAD)
        enc_padding_mask = tf.reshape(enc_padding_mask,shape=[inp.shape[0],1,1,self.config.max_turn,self.config.max_utterance_len])
        inp = tf.reshape(inp, shape=[inp.shape[0],self.config.max_turn,self.config.max_utterance_len])
        utterance=[]
        for utt,pad_mask in zip(tf.unstack(inp,axis=1),tf.unstack(enc_padding_mask,axis=-2)):
            enc_output = self.encoder1(self.convert2embed(utt,positions=self.tokens_encoding), training, pad_mask)  # (batch_size, inp_seq_len, d_model)
            enc_output += (tf.reshape(pad_mask, shape=[pad_mask.shape[0],pad_mask.shape[-1],1]) * -1e9)
            enc_output = tf.reduce_sum(enc_output, axis=1)
            utterance.append(enc_output)
        utterance = tf.stack(utterance,axis=1)
        enc1_padding_mask = tf.cast(tf.reduce_sum(1-enc_padding_mask,axis=-1),tf.bool)#[batch,1,1,max_turn]
        enc1_padding_mask = 1 - tf.cast(enc1_padding_mask,dtype=utterance.dtype)
        utterance += self.turns_encoding[:, :utterance.shape[1], :]
        enc_output = self.encoder2(utterance,training,enc1_padding_mask)
        dec_padding_mask = enc1_padding_mask
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output = self.decoder(self.convert2embed(tar[:,:-1],positions=self.tokens_encoding), enc_output, training, look_ahead_mask, dec_padding_mask,cache=None)
        final_output = self.outputLayer(dec_output)
#         final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        loss = self.loss_function(tar[:,1:], final_output)
        return final_output, loss
    def BeamDecoder(self,features, training):
        inp = features["src"]
        batch_size = inp.shape[0]
        enc_padding_mask = create_padding_mask(inp,padId=self.PAD)
        enc_padding_mask = tf.reshape(enc_padding_mask,shape=[inp.shape[0],1,1,self.config.max_turn,self.config.max_utterance_len])
        inp = tf.reshape(inp, shape=[inp.shape[0],self.config.max_turn,self.config.max_utterance_len])
        utterance=[]
        for utt,pad_mask in zip(tf.unstack(inp,axis=1),tf.unstack(enc_padding_mask,axis=-2)):
            enc_output = self.encoder1(self.convert2embed(utt), training, pad_mask)  # (batch_size, inp_seq_len, d_model)
            enc_output += (tf.reshape(pad_mask, shape=[pad_mask.shape[0],pad_mask.shape[-1],1]) * -1e9)
            enc_output = tf.reduce_mean(enc_output, axis=1)
            utterance.append(enc_output)
        utterance = tf.stack(utterance,axis=1)
        enc1_padding_mask = tf.cast(tf.reduce_sum(1-enc_padding_mask,axis=-1),tf.bool)#[batch,1,1,max_turn]
        enc1_padding_mask = 1 - tf.cast(enc1_padding_mask,dtype=utterance.dtype)
        utterance += self.tokens_encoding[:, :utterance.shape[1], :]
        enc_output = self.encoder2(utterance,training,enc1_padding_mask)
        look_ahead_mask = create_look_ahead_mask(self.config.decode_length)
        startIdx = [self.SOS] * batch_size
        cache = {
            str(i):{
                "k":self.decoder.getDecoderLayer(i).GetSelfAtt().split_heads(tf.zeros([batch_size, 0, enc_output.shape[-1]]), batch_size),
                  "v":self.decoder.getDecoderLayer(i).GetSelfAtt().split_heads(tf.zeros([batch_size, 0, enc_output.shape[-1]]), batch_size)
                  }for i in range(self.config.decoder_layers)}
        for i in range(self.config.decoder_layers):
            CrossObj = self.decoder.getDecoderLayer(i).GetCrossfAtt()
            k = CrossObj.wk(enc_output) # (batch_size, seq_len, d_model)
            v = CrossObj.wv(enc_output) # (batch_size, seq_len, d_model)
            k = CrossObj.split_heads(k, enc_output.shape[0])  # (batch_size, num_heads, seq_len_k, depth)
            v = CrossObj.split_heads(v, enc_output.shape[0])  # (batch_size, num_heads, seq_len_v, depth)
            cache[str(i)]["k_encdec"] = k
            cache[str(i)]["v_encdec"] = v
        cache["enc_output"] = enc_output
        cache["dec_padding_mask"] = enc1_padding_mask
        def symbols_to_logits_fn(tgtids,i,cache):
            """
                tgtids:batch * seq
            """
            dec_input = self.embedding(tgtids[:,-1:])
            dec_input = dec_input + self.tokens_encoding[:, i:i + 1, :]
            look_ahead_mask_t = look_ahead_mask[i:i + 1, :i + 1]
            dec_target_padding_mask = create_padding_mask(tgtids[:,-1:],padId=self.PAD)
            combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask_t)
            # dec_output.shape == (batch_size, tar_seq_len, d_model)
            dec_output = self.decoder(dec_input, 
                                      cache["enc_output"], 
                                      training, 
                                      combined_mask, 
                                      cache["dec_padding_mask"],
                                      cache = cache )
            final_output = self.outputLayer(dec_output)
#             final_output = self.final_layer(dec_output)# (batch_size, tar_seq_len, target_vocab_size
            logits = final_output[:,-1,:]
            return logits,cache
        ids, scores=beam_search(symbols_to_logits_fn=symbols_to_logits_fn,
                initial_ids=startIdx,
                beam_size=self.config.beam_size,
                decode_length=self.config.decode_length,
                vocab_size=self.output_size,
                alpha=self.config.alpha,
                states=cache,
                eos_id=self.EOS,
                stop_early=True)
        
        return ids[:,0,:]

class ConTransformer(tf.keras.Model):
    def __init__(self,vocab_size,matrix,config,PROM=0,EOS=0,PAD=0,rate=0.1):
        super(ConTransformer, self).__init__()
        self.SOS = PROM
        self.EOS = EOS
        self.PAD = PAD
        self.config = config
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                   config.emb_size,
                                                   embeddings_initializer=keras.initializers.constant(matrix),
                                                   trainable=True)
        self.tokens_encoding = positional_encoding(config.max_utterance_len*config.max_turn, config.d_model)
        
        self.encoder = Encoder(num_layers=config.num_layers, 
                               d_model=config.d_model, 
                               num_heads=config.num_heads, 
                               dff=config.dff,
                               rate=rate)
        self.decoder = Decoder(num_layers=config.decoder_layers, 
                               d_model=config.d_model,
                               num_heads=config.num_heads,
                               dff=config.dff,
                               rate=rate)
        self.output_size = vocab_size
    def convert2embed(self,x):
        seq_len = x.shape[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.config.d_model, tf.float32))
        x += self.tokens_encoding[:, :seq_len, :]
        return x
    def getEmbeddingTable(self):
        return self.embedding.embeddings
    def outputLayer(self,logits):
        """
        parameters:
            logits:batch * seq * d
        return: batch  * seq * vocabSize
        """
        return tf.einsum("bsd,vd->bsv",logits,self.getEmbeddingTable())
    def loss_function(self,real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, self.PAD))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    @tf.function
    def call(self,features,training):
        inp = features["src"]
        tar = features["tgt"]
        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp,tar[:,:-1],padId=self.PAD)
        enc_output = self.encoder(self.convert2embed(inp), training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output = self.decoder(self.convert2embed(tar[:,:-1]), enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.outputLayer(dec_output)
#         final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        loss = self.loss_function(tar[:,1:], final_output)
        return final_output, loss
    def BeamDecoder(self,features, training):
        inp = features["src"]
        batch_size = inp.shape[0]
        enc_padding_mask = create_padding_mask(inp,padId=self.PAD)
        enc_output = self.encoder(self.convert2embed(inp), training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        look_ahead_mask = create_look_ahead_mask(self.config.decode_length)
        startIdx = [self.SOS] * batch_size
        cache = {
            str(i):{
                "k":self.decoder.getDecoderLayer(i).GetSelfAtt().split_heads(tf.zeros([batch_size, 0, enc_output.shape[-1]]), batch_size),
                  "v":self.decoder.getDecoderLayer(i).GetSelfAtt().split_heads(tf.zeros([batch_size, 0, enc_output.shape[-1]]), batch_size)
                  }for i in range(self.config.decoder_layers)}
        for i in range(self.config.decoder_layers):
            CrossObj = self.decoder.getDecoderLayer(i).GetCrossfAtt()
            k = CrossObj.wk(enc_output) # (batch_size, seq_len, d_model)
            v = CrossObj.wv(enc_output) # (batch_size, seq_len, d_model)
            k = CrossObj.split_heads(k, enc_output.shape[0])  # (batch_size, num_heads, seq_len_k, depth)
            v = CrossObj.split_heads(v, enc_output.shape[0])  # (batch_size, num_heads, seq_len_v, depth)
            cache[str(i)]["k_encdec"] = k
            cache[str(i)]["v_encdec"] = v
        cache["enc_output"] = enc_output
        cache["dec_padding_mask"] = enc_padding_mask
        def symbols_to_logits_fn(tgtids,i,cache):
            """
                tgtids:batch * seq
            """
            dec_input = self.embedding(tgtids[:,-1:])
            dec_input = dec_input + self.tokens_encoding[:, i:i + 1, :]
            look_ahead_mask_t = look_ahead_mask[i:i + 1, :i + 1]
            dec_target_padding_mask = create_padding_mask(tgtids[:,-1:],padId=self.PAD)
            combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask_t)
            # dec_output.shape == (batch_size, tar_seq_len, d_model)
            dec_output = self.decoder(dec_input, 
                                      cache["enc_output"],
                                      training, 
                                      combined_mask, 
                                      cache["dec_padding_mask"],
                                      cache = cache)
            final_output = self.outputLayer(dec_output)
#             final_output = self.final_layer(dec_output)# (batch_size, tar_seq_len, target_vocab_size
            logits = final_output[:,-1,:]
            return logits,cache
        ids, scores=beam_search(symbols_to_logits_fn=symbols_to_logits_fn,
                initial_ids=startIdx,
                beam_size=self.config.beam_size,
                decode_length=self.config.decode_length,
                vocab_size=self.output_size,
                alpha=self.config.alpha,
                states=cache,
                eos_id=self.EOS,
                stop_early=True)
        
        return ids[:,0,:]
        
        
        
        
        
        
        
        
        
    
