import re
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from vietnamese_processor import VietnameseTextPreprocessor

    
class VLSP2018Loader:
    
    @staticmethod
    def load(train_csv_path, val_csv_path, test_csv_path):
        dataset_paths = {'train': train_csv_path, 'val': val_csv_path, 'test': test_csv_path}
        raw_datasets = load_dataset('csv', data_files={ k: v for k, v in dataset_paths.items() if v })
        return raw_datasets.map(VLSP2018Loader.aspect_categories_to_onehot)
    
    @staticmethod
    def aspect_categories_to_onehot(review):
        polarity_one_hot = {
            0: [1, 0, 0, 0], # None
            1: [0, 1, 0, 0], # Positive
            2: [0, 0, 1, 0], # Negative
            3: [0, 0, 0, 1], # Neutral
        }
        aspect_categories = list(review)[1:]
        for aspect_category in aspect_categories:
            review[aspect_category] = polarity_one_hot[review[aspect_category]]
        return review
    
    @staticmethod
    def preprocess(raw_datasets, preprocessor, tokenizer, max_length):
        def tokenize_fn(review):
            review['Review'] = preprocessor.process_text(review['Review'])
            return tokenizer(review['Review'], max_length=max_length, padding='max_length', truncation=True)
        return raw_datasets.map(tokenize_fn)
                

class VLSP2018Parser:
    def __init__(self, train_txt_path, val_txt_path=None, test_txt_path=None):
        self.dataset_paths = { 'train': train_txt_path, 'val': val_txt_path, 'test': test_txt_path }
        self.reviews = { 'train': [], 'val': [], 'test': [] }
        self.polarity_mapping = {'positive': 1, 'negative': 2, 'neutral': 3}
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
                        review_data[aspect_category] = self.polarity_mapping[polarity.strip()]
                    
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