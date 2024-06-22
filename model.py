import numpy as np
from matplotlib import pyplot as plt
from processors.vlsp2018_processor import PolarityMapping

from transformers import TFAutoModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate


def vlsp2018_multitask_model(pretrained_huggingface_name, max_length, aspect_category_names, optimizer, multi_branch=False):
    # https://riccardo-cantini.netlify.app/post/bert_text_classification
    inputs = {
        'input_ids'     : Input((max_length,), dtype='int32', name='input_ids'), 
        'token_type_ids': Input((max_length,), dtype='int32', name='token_type_ids'), 
        'attention_mask': Input((max_length,), dtype='int32', name='attention_mask'),
    }
    pretrained_bert = TFAutoModel.from_pretrained(pretrained_huggingface_name, output_hidden_states=True)
    hidden_states = pretrained_bert(inputs).hidden_states

    # https://github.com/huggingface/transformers/issues/1328
    pooled_output = concatenate(
        tuple([hidden_states[i] for i in range(-4, 0)]), 
        name='last_4_hidden_states', axis=-1
    )
    
    x = Dropout(0.2)(pooled_output[:, 0, :])
    outputs = [
        Dense(4, activation='softmax', name=label.replace('#', '-').replace('&', '_'))(x) 
        for label in aspect_category_names
    ]
    
    if not multi_branch: 
        outputs = concatenate(outputs, name='FlattenOneHotLabels')
        loss = 'binary_crossentropy'
    else: loss = 'sparse_categorical_crossentropy'
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss)
    return model
    

def plot_training_history(history, figsize=(15, 5)):
    plt.figure(figsize=figsize)
    plt.plot(history['loss'], linestyle='solid', marker='o', color='crimson', label='Train')
    plt.plot(history['val_loss'], linestyle='solid', marker='o', color='dodgerblue', label='Validation')
    plt.xlabel('Epochs', fontsize = 14)
    plt.ylabel('Loss',fontsize=14)
    plt.title('Loss', fontsize=15)
    plt.legend(loc='best')
    plt.show()
    
    
def acsa_predict(model, text_data, batch_size=1, multi_branch=False):
    y_pred = model.predict(text_data, batch_size=batch_size, verbose=1)
    if not multi_branch: 
        y_pred = y_pred.reshape(len(y_pred), -1, 4)
        return np.argmax(y_pred, axis=-1)
    return np.argmax(y_pred, axis=-1).T


def print_acsa_pred(aspect_category_names, y_pred):
    polarities = map(lambda x: PolarityMapping.INDEX_TO_POLARITY[x], y_pred)
    for aspect_category, polarity in zip(aspect_category_names, polarities): 
        if polarity: print(f'=> {aspect_category},{polarity}')