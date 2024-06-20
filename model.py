from transformers import TFAutoModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate


def vlsp2018_multitask_model(huggingface_name, max_length, aspect_category_names, multibranch=False):
    # https://riccardo-cantini.netlify.app/post/bert_text_classification
    inputs = {
        'input_ids'     : Input((max_length,), dtype='int32', name='input_ids'), 
        'token_type_ids': Input((max_length,), dtype='int32', name='token_type_ids'), 
        'attention_mask': Input((max_length,), dtype='int32', name='attention_mask'),
    }
    pretrained_bert = TFAutoModel.from_pretrained(huggingface_name, output_hidden_states=True)
    hidden_states = pretrained_bert(inputs).hidden_states

    # https://github.com/huggingface/transformers/issues/1328
    pooled_output = concatenate(
        tuple([hidden_states[i] for i in range(-4, 0)]), 
        name='last_4_hidden_states', axis=-1
    )[:, 0, :]
    x = Dropout(0.2)(pooled_output)

    outputs = [
        Dense(4, activation='softmax', name=label.replace('#', '-').replace('&', '_'))(x) 
        for label in aspect_category_names
    ]
    if not multibranch: outputs = concatenate(outputs, name='flat1hot_aspect_cate')
    return Model(inputs=inputs, outputs=outputs)