import tensorflow as tf
import tensorflow_addons as tfa

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time
import random as biubiubiu
import base64
import struct

import SignalSimulatorTools as sst


# global variables
start_token_as_i_see = '<start>' 
end_token_as_i_see = '<end>'
# max symbol number
min_symbol_num = 3
max_symbol_num = 10
data_set_num = 10240



def download_nmt():
    path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

    path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"
    return path_to_file


def generate_signal(mode = "modulated signal"):
    print("YOU ARE USING <%s>!!!!"%mode)

    signal_list = []
    symbol_list = []

    transer = sst.transmitter()
    transer.set_carrier_frequency(2500000)
    transer.set_filter_span(16)
    transer.set_modulation_type("QPSK")
    transer.set_oversamping_factor(4)
    transer.set_roll_ratio(0.5) # if not set, will generate randomly in [0.1 ,0.5)
    transer.set_snr(2.0) # if not set, will generate randomly in [-2 ,8)dB
    transer.init_setting()
    for _ in range(data_set_num):
        transer.generate_signal_by_symbol_num(symbols_num = biubiubiu.randint(min_symbol_num, max_symbol_num))
        modulated_signal = transer.get_modulated_signal()
        symbols = transer.get_symbols()

        signal_list.append(modulated_signal)
        symbol_list.append(symbols)

    return signal_list, symbol_list

def create_dateset(signal_list, symbol_list):
    res1 = []
    res2 = []
    for idx, signal_bar in enumerate(signal_list):
        res = ""
        for item in signal_bar:
            item = struct.pack("d", item)
            item = base64.encodebytes(item).decode()
            item = re.sub(r"[^0-9a-zA-Z+/]+", " ", item)
            for i in range(0,len(item),2):
                end = min([i+2, len(item)])
                res += item[i:end] + " "
        res1.append(res)

    for idx, symbols in enumerate(symbol_list):
        for idxx, item in enumerate(symbols):
            symbols[idxx] = str(item)
        res2.append(" ".join(symbols))

    f = open(".\\signal_record_train.txt", 'w')
    for item1, item2 in zip(res1, res2):
        f.write(item1+"\t"+item2+"\n")
    f.close()
    return res1, res2

signal_list, symbol_list = generate_signal()
str_inp, str_label = create_dateset(signal_list, symbol_list)
a, b = signal_list, symbol_list

class NMTDataset:
    def __init__(self, problem_type='en-spa'):
        self.problem_type = 'en-spa'
        self.inp_lang_tokenizer = None
        self.targ_lang_tokenizer = None


    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    ## Step 1 and Step 2 
    def preprocess_sentence(self, w):
        #w = self.unicode_to_ascii(w.lower().strip())
        w = self.unicode_to_ascii(w.strip())

        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)

        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        #w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
        w = re.sub(r"[^0-9a-zA-Z+/]+", " ", w)

        w = w.strip()

        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        #w = '<start> ' + w + ' <end>'
        w = '%s '%start_token_as_i_see + w + ' %s'%end_token_as_i_see
        return w

    def create_dataset(self, path, num_examples):
        # path : path to spa-eng.txt file
        # num_examples : Limit the total number of training example for faster training (set num_examples = len(lines) to use full data)
        lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
        word_pairs = [[self.preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

        return zip(*word_pairs)

    # Step 3 and Step 4
    def tokenize(self, lang):
        # lang = list of sentences in a language

        # print(len(lang), "example sentence: {}".format(lang[0]))
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>', lower=False)
        lang_tokenizer.fit_on_texts(lang)

        ## tf.keras.preprocessing.text.Tokenizer.texts_to_sequences converts string (w1, w2, w3, ......, wn) 
        ## to a list of correspoding integer ids of words (id_w1, id_w2, id_w3, ...., id_wn)
        tensor = lang_tokenizer.texts_to_sequences(lang) 

        ## tf.keras.preprocessing.sequence.pad_sequences takes argument a list of integer id sequences 
        ## and pads the sequences to match the longest sequences in the given input
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

        return tensor, lang_tokenizer

    def load_dataset(self, path=None, num_examples=None):
        # creating cleaned input, output pairs
        signal_list, symbol_list = generate_signal()
        str_inp, str_label = create_dateset(signal_list, symbol_list)
        str_inp, str_label = self.create_dataset(".\\signal_record.txt", data_set_num)
        targ_lang, inp_lang = str_label, str_inp

        input_tensor, inp_lang_tokenizer = self.tokenize(inp_lang)
        target_tensor, targ_lang_tokenizer = self.tokenize(targ_lang)

        return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

    def call(self, num_examples, BUFFER_SIZE, BATCH_SIZE):
        #file_path = download_nmt()
        #input_tensor, target_tensor, self.inp_lang_tokenizer, self.targ_lang_tokenizer = self.load_dataset(file_path, num_examples)
        input_tensor, target_tensor, self.inp_lang_tokenizer, self.targ_lang_tokenizer = self.load_dataset()

        input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

        train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
        val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

        return train_dataset, val_dataset, self.inp_lang_tokenizer, self.targ_lang_tokenizer


BUFFER_SIZE = 32000
BATCH_SIZE = 64
# Let's limit the #training examples for faster training
num_examples = data_set_num

dataset_creator = NMTDataset('en-spa')
train_dataset, val_dataset, inp_lang, targ_lang = dataset_creator.call(num_examples, BUFFER_SIZE, BATCH_SIZE)

example_input_batch, example_target_batch = next(iter(train_dataset))
example_input_batch.shape, example_target_batch.shape

vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
max_length_input = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]

embedding_dim = 128
units = 128
steps_per_epoch = num_examples//BATCH_SIZE

print("max_length_english, max_length_spanish, vocab_size_english, vocab_size_spanish")
max_length_input, max_length_output, vocab_inp_size, vocab_tar_size


##### 

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    ##-------- LSTM layer in Encoder ------- ##
    self.lstm_layer = tf.keras.layers.LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform',
                                   dropout = 0.2)



  def call(self, x, hidden, training = False):
    x = self.embedding(x)
    output, h, c = self.lstm_layer(x, initial_state = hidden, training = training)
    return output, h, c

  def initialize_hidden_state(self):
    return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]



## Test Encoder Stack

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)


# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_h, sample_c = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder h vecotr shape: (batch size, units) {}'.format(sample_h.shape))
print ('Encoder c vector shape: (batch size, units) {}'.format(sample_c.shape))



class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, attention_type='luong'):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.attention_type = attention_type

    # Embedding Layer
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    #Final Dense layer on which softmax will be applied
    self.fc = tf.keras.layers.Dense(vocab_size)

    # Define the fundamental cell for decoder recurrent structure
    self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units, dropout = 0.2)
    #self.decoder_rnn_cell = tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(self.dec_units), tf.keras.layers.LSTMCell(self.dec_units)])



    # Sampler
    self.sampler = tfa.seq2seq.sampler.TrainingSampler()

    # Create attention mechanism with memory = None
    self.attention_mechanism = self.build_attention_mechanism(self.dec_units, 
                                                              None, self.batch_sz*[max_length_input], self.attention_type)

    # Wrap attention mechanism with the fundamental rnn cell of decoder
    self.rnn_cell = self.build_rnn_cell(batch_sz)

    # Define the decoder with respect to fundamental rnn cell
    self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)
    #self.decoder = tfa.seq2seq.BasicDecoder(self.decoder_rnn_cell, sampler=self.sampler, output_layer=self.fc)

  def build_rnn_cell(self, batch_sz):
    rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell, 
                                  self.attention_mechanism, attention_layer_size=self.dec_units)
    return rnn_cell

  def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type='luong'):
    # ------------- #
    # typ: Which sort of attention (Bahdanau, Luong)
    # dec_units: final dimension of attention outputs 
    # memory: encoder hidden states of shape (batch_size, max_length_input, enc_units)
    # memory_sequence_length: 1d array of shape (batch_size) with every element set to max_length_input (for masking purpose)

    if(attention_type=='bahdanau'):
      return tfa.seq2seq.BahdanauAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)
    else:
      return tfa.seq2seq.LuongAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)

  def build_initial_state(self, batch_sz, encoder_state, Dtype):
    decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_sz, dtype=Dtype)
    #decoder_initial_state = self.decoder_rnn_cell.get_initial_state(inputs = encoder_state, batch_size=batch_sz, dtype=Dtype)
    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
    return decoder_initial_state


  def call(self, inputs, initial_state, training = False):
    x = self.embedding(inputs)
    outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_sz*[max_length_output-1], training = training)
    return outputs


# Test decoder stack

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, 'luong')
sample_x = tf.random.uniform((BATCH_SIZE, max_length_output))
decoder.attention_mechanism.setup_memory(sample_output)
initial_state = decoder.build_initial_state(BATCH_SIZE, [sample_h, sample_c], tf.float32)


sample_decoder_outputs = decoder(sample_x, initial_state)

print("Decoder Outputs Shape: ", sample_decoder_outputs.rnn_output.shape)


learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1280,
    decay_rate=0.5)
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)


def loss_function(real, pred):
  # real shape = (BATCH_SIZE, max_length_output)
  # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
  cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
  loss = cross_entropy(y_true=real, y_pred=pred)
  mask = tf.logical_not(tf.math.equal(real,0))   #output 0 for y=0 else output 1
  mask = tf.cast(mask, dtype=loss.dtype)  
  loss = mask* loss
  loss = tf.reduce_mean(loss)
  return loss


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)




@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_h, enc_c = encoder(inp, enc_hidden, training = True)


    dec_input = targ[ : , :-1 ] # Ignore <end> token
    real = targ[ : , 1: ]         # ignore <start> token

    # Set the AttentionMechanism object with encoder_outputs
    decoder.attention_mechanism.setup_memory(enc_output)

    # Create AttentionWrapperState as initial_state for decoder
    decoder_initial_state = decoder.build_initial_state(BATCH_SIZE, [enc_h, enc_c], tf.float32)
    pred = decoder(dec_input, decoder_initial_state, training = True)
    logits = pred.rnn_output
    loss = loss_function(real, logits)

  variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))

  return loss


EPOCHS = 1

for epoch in range(EPOCHS):
  start = time.time()

  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0
  # print(enc_hidden[0].shape, enc_hidden[1].shape)

  for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):
    batch_loss = train_step(inp, targ, enc_hidden)
    total_loss += batch_loss

    if batch % 100 == 0:
      print('epoch {} batch {} loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
  # saving (checkpoint) the model every 2 epochs
  if (epoch + 1) % 1 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

  print('epoch {} loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
  print('time taken for 1 epoch {} sec\n'.format(time.time() - start))




#def evaluate_sentence(sentence):
#  sentence = dataset_creator.preprocess_sentence(sentence)

#  inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
#  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
#                                                          maxlen=max_length_input,
#                                                          padding='post')
#  inputs = tf.convert_to_tensor(inputs)
#  inference_batch_size = inputs.shape[0]
#  result = ''

#  enc_start_state = [tf.zeros((inference_batch_size, units)), tf.zeros((inference_batch_size,units))]
#  enc_out, enc_h, enc_c = encoder(inputs, enc_start_state)

#  dec_h = enc_h
#  dec_c = enc_c

#  start_tokens = tf.fill([inference_batch_size], targ_lang.word_index[start_token_as_i_see])
#  end_token = targ_lang.word_index[end_token_as_i_see]

#  greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

#  # Instantiate BasicDecoder object
#  decoder_instance = tfa.seq2seq.BasicDecoder(cell=decoder.rnn_cell, sampler=greedy_sampler, output_layer=decoder.fc)
#  # Setup Memory in decoder stack
#  decoder.attention_mechanism.setup_memory(enc_out)

#  # set decoder_initial_state
#  decoder_initial_state = decoder.build_initial_state(inference_batch_size, [enc_h, enc_c], tf.float32)


#  ### Since the BasicDecoder wraps around Decoder's rnn cell only, you have to ensure that the inputs to BasicDecoder 
#  ### decoding step is output of embedding layer. tfa.seq2seq.GreedyEmbeddingSampler() takes care of this. 
#  ### You only need to get the weights of embedding layer, which can be done by decoder.embedding.variables[0] and pass this callabble to BasicDecoder's call() function

#  decoder_embedding_matrix = decoder.embedding.variables[0]

#  outputs, _, _ = decoder_instance(decoder_embedding_matrix, start_tokens = start_tokens, end_token= end_token, initial_state=decoder_initial_state)
#  return outputs.sample_id.numpy()

##def translate(sentence):
##  result = evaluate_sentence(sentence)
##  print(result)
##  result = targ_lang.sequences_to_texts(result)
##  print('Input: %s' % (sentence))
##  print('Predicted translation: {}'.format(result))

##  return result[0]

##  # restoring the latest checkpoint in checkpoint_dir
##checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


## test here
#with open("signal_record.txt") as f:
#    f_content = f.read()

#sentence_list = f_content.split("\n")
#label_list = []
#for idx, item in enumerate(sentence_list):
#    if item == "":
#        continue
#    sentence_here, label_here = item.split("\t")
#    sentence_list[idx] = sentence_here
#    label_list.append(label_here)

#def check_sentence(sx, sy):
#    sx = sx.split()
#    sy = sy.split()

#    L = min([len(sx), len(sy)])
#    count = 0 
#    for i in range(L):
#        if sx[i] != sy[i]:
#            count += 1
#    count += max([len(sx), len(sy)]) - L

#    return count

#accuracy = 0
#length = 0
#for sentence, label in zip(sentence_list, label_list):
#    res = translate(sentence)
#    res = res.replace(" "+end_token_as_i_see, "")
#    count_tmp =  check_sentence(res, label)
#    length += len(label.split())
#    accuracy += count_tmp
#    print("count_tmp is %d"%count_tmp)
#    print("length is %d"%length)

#print("BER is %f\%."%(accuracy / length * 100))