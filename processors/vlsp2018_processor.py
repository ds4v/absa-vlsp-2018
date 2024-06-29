import re
import csv
from tqdm import tqdm
from datasets import load_dataset


class PolarityMapping:
    INDEX_TO_POLARITY = { 0: None, 1: 'positive', 2: 'negative', 3: 'neutral' }
    # INDEX_TO_ONEHOT = { i: [1 if i == j else 0 for j in INDEX_TO_POLARITY] for i in INDEX_TO_POLARITY } 
    # POLARITY_TO_INDEX = { polarity: index for index, polarity in INDEX_TO_POLARITY.items() }
    INDEX_TO_ONEHOT = { 0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0], 3: [0, 0, 0, 1] }
    POLARITY_TO_INDEX = { None: 0, 'positive': 1, 'negative': 2, 'neutral': 3 }

    
class VLSP2018Loader:
    
    @staticmethod
    def load(train_csv_path, val_csv_path, test_csv_path):
        dataset_paths = {'train': train_csv_path, 'val': val_csv_path, 'test': test_csv_path}
        raw_datasets = load_dataset('csv', data_files={ k: v for k, v in dataset_paths.items() if v })
        return raw_datasets
          
          
    @staticmethod
    def preprocess_and_tokenize(text_data, preprocessor, tokenizer, batch_size, max_length):
        print('[INFO] Preprocessing and tokenizing text data...')
        def transform_each_batch(batch):
            preprocessed_batch = preprocessor.process_batch(batch)
            return tokenizer(preprocessed_batch, max_length=max_length, padding='max_length', truncation=True)
        
        if type(text_data) == str: return transform_each_batch([text_data])
        return text_data.map(
            lambda reviews: transform_each_batch(reviews['Review']), 
            batched=True, batch_size=batch_size
        ).remove_columns('Review')
    
    
    @staticmethod
    def labels_to_flatten_onehot(datasets):
        print('[INFO] Transforming "Aspect#Categoy,Polarity" labels to flattened one-hot encoding...')
        model_input_names = ['input_ids', 'token_type_ids', 'attention_mask']
        label_columns = [col for col in datasets['train'].column_names if col not in ['Review', *model_input_names]]
        def transform_each_review(review): # Convert each Aspect#Categoy,Polarity to one-hot encoding and merge them into 1D list
            review['FlattenOneHotLabels'] = sum([
                PolarityMapping.INDEX_TO_ONEHOT[review[aspect_category]] # Get one-hot encoding
                for aspect_category in label_columns
            ], []) # Need to be flattened to match the model's output shape
            return review 
        return datasets.map(transform_each_review, num_proc=8).select_columns(['FlattenOneHotLabels', *model_input_names])


class VLSP2018Parser:
    def __init__(self, train_txt_path, val_txt_path=None, test_txt_path=None):
        self.dataset_paths = { 'train': train_txt_path, 'val': val_txt_path, 'test': test_txt_path }
        self.reviews = { 'train': [], 'val': [], 'test': [] }
        self.aspect_categories = set()
        
        for dataset_type, txt_path in self.dataset_paths.items():
            if not txt_path: 
                self.dataset_paths.pop(dataset_type)
                self.reviews.pop(dataset_type)
        self._parse_input_files()


    def _parse_input_files(self):
        print(f'[INFO] Parsing {len(self.dataset_paths)} input files...')
        for dataset_type, txt_path in self.dataset_paths.items():
            with open(txt_path, 'r', encoding='utf-8') as txt_file:
                content = txt_file.read()
                review_blocks = content.strip().split('\n\n')
                
                for block in tqdm(review_blocks):
                    lines = block.split('\n')
                    sentiment_info = re.findall(r'\{([^,]+)#([^,]+), ([^}]+)\}', lines[2].strip())

                    review_data = {}
                    for aspect, category, polarity in sentiment_info:
                        aspect_category = f'{aspect.strip()}#{category.strip()}'
                        self.aspect_categories.add(aspect_category)
                        review_data[aspect_category] = PolarityMapping.POLARITY_TO_INDEX[polarity.strip()]
                    
                    self.reviews[dataset_type].append((lines[1].strip(), review_data))
        self.aspect_categories = sorted(self.aspect_categories)


    def txt2csv(self):
        print('[INFO] Converting parsed data to CSV files...')
        for dataset, txt_path in self.dataset_paths.items():
            csv_path = txt_path.replace('.txt', '.csv')
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['Review'] + self.aspect_categories)

                for review_text, review_data in tqdm(self.reviews[dataset]):
                    row = [review_text] + [review_data.get(aspect_category, 0) for aspect_category in self.aspect_categories]
                    writer.writerow(row)
    
    @staticmethod
    def vlsp_save_as(save_path, raw_texts, encoded_review_labels, aspect_category_names):
        with open(save_path, 'w', encoding='utf-8') as file:
            for index, encoded_label in tqdm(enumerate(encoded_review_labels)):
                polarities = map(lambda x: PolarityMapping.INDEX_TO_POLARITY[x], encoded_label)
                acsa = ', '.join(
                    f'{{{aspect_category}, {polarity}}}' 
                    for aspect_category, polarity in zip(aspect_category_names, polarities) if polarity
                )
                file.write(f"#{index + 1}\n{raw_texts[index]}\n{acsa}\n\n")
                    

if __name__ == '__main__':
    # Hotel Domain
    hotel_train_path = 'datasets/vlsp2018_hotel/1-VLSP2018-SA-Hotel-train.txt'
    hotel_val_path = 'datasets/vlsp2018_hotel/2-VLSP2018-SA-Hotel-dev.txt'
    hotel_test_path = 'datasets/vlsp2018_hotel/3-VLSP2018-SA-Hotel-test.txt'
    vlsp_hotel_parser = VLSP2018Parser(hotel_train_path, hotel_val_path, hotel_test_path)
    vlsp_hotel_parser.txt2csv()
    
    # Restaurant Domain
    restaurant_train_path = 'datasets/vlsp2018_restaurant/1-VLSP2018-SA-Restaurant-train.txt'
    restaurant_val_path = 'datasets/vlsp2018_restaurant/2-VLSP2018-SA-Restaurant-dev.txt'
    restaurant_test_path = 'datasets/vlsp2018_restaurant/3-VLSP2018-SA-Restaurant-test.txt'
    vlsp_restaurant_parser = VLSP2018Parser(restaurant_train_path, restaurant_val_path, restaurant_test_path)
    vlsp_restaurant_parser.txt2csv()