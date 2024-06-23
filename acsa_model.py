import numpy as np
from matplotlib import pyplot as plt
from processors.vlsp2018_processor import PolarityMapping

from transformers import TFAutoModel
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate


class VLSP2018MultiTask(Model):
    def __init__(self, pretrained_huggingface_name, aspect_category_names, optimizer, multi_branch=False, **kwargs):
        super(VLSP2018MultiTask, self).__init__(**kwargs)
        self.aspect_category_names = aspect_category_names
        self.multi_branch = multi_branch

        self.pretrained_bert = TFAutoModel.from_pretrained(pretrained_huggingface_name, output_hidden_states=True)
        self.last_4_hidden_states = Concatenate(name='last_4_hidden_states')
        self.dropout = Dropout(0.2)
        self.dense_layers = [
            Dense(4, activation='softmax', name=label.replace('#', '-').replace('&', '_')) 
            for label in self.aspect_category_names
        ]

        if self.multi_branch: self.loss = 'sparse_categorical_crossentropy'
        else: 
            self.flatten_onehot_labels = Concatenate(name='FlattenOneHotLabels')
            self.loss = 'binary_crossentropy'
        self.compile(optimizer=optimizer, loss=self.loss)
    

    def call(self, inputs):
        # https://riccardo-cantini.netlify.app/post/bert_text_classification
        # https://github.com/huggingface/transformers/issues/1328
        hidden_states = self.pretrained_bert(inputs).hidden_states 
        pooled_output = self.last_4_hidden_states(tuple([hidden_states[i] for i in range(-4, 0)])) 
        x = self.dropout(pooled_output[:, 0, :]) 
        
        outputs = [dense_layer(x) for dense_layer in self.dense_layers]
        if self.multi_branch: return outputs
        return self.flatten_onehot_labels(outputs)
    
    
    def acsa_predict(self, text_data, batch_size=1):
        y_pred = self.predict(text_data, batch_size=batch_size, verbose=1)
        if not self.multi_branch: 
            y_pred = y_pred.reshape(len(y_pred), -1, 4)
            return np.argmax(y_pred, axis=-1)
        return np.argmax(y_pred, axis=-1).T


    def print_acsa_pred(self, y_pred):
        polarities = map(lambda x: PolarityMapping.INDEX_TO_POLARITY[x], y_pred)
        for aspect_category, polarity in zip(self.aspect_category_names, polarities): 
            if polarity: print(f'=> {aspect_category},{polarity}')
    

def plot_training_history(history, figsize=(15, 5)):
    plt.figure(figsize=figsize)
    plt.plot(history['loss'], linestyle='solid', marker='o', color='crimson', label='Train')
    plt.plot(history['val_loss'], linestyle='solid', marker='o', color='dodgerblue', label='Validation')
    plt.xlabel('Epochs', fontsize = 14)
    plt.ylabel('Loss',fontsize=14)
    plt.title('Loss', fontsize=15)
    plt.legend(loc='best')
    plt.show()