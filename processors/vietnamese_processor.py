# https://github.com/behitek/text-classification-tutorial/blob/master/text_classification_tutorial.ipynb
import os
import emoji
import urllib
import requests
import regex as re

from io import StringIO
from vncorenlp import VnCoreNLP
from transformers import pipeline


class VietnameseTextCleaner: # https://ihateregex.io
    VN_CHARS = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÍÌỈĨỊÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ'
    
    @staticmethod
    def remove_html(text):
        return re.sub(r'<[^>]*>', '', text)
    
    @staticmethod
    def remove_emoji(text):
        return emoji.replace_emoji(text, '')
    
    @staticmethod
    def remove_url(text):
        return re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()!@:%_\+.~#?&\/\/=]*)', '', text)
    
    @staticmethod
    def remove_email(text):
        return re.sub(r'[^@ \t\r\n]+@[^@ \t\r\n]+\.[^@ \t\r\n]+', '', text)
    
    @staticmethod
    def remove_phone_number(text):
        return re.sub(r'^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$', '', text)
    
    @staticmethod
    def remove_hashtags(text):
        return re.sub(r'#\w+', '', text)
    
    @staticmethod
    def remove_unnecessary_characters(text):
        text = re.sub(fr"[^\sa-zA-Z0-9{VietnameseTextCleaner.VN_CHARS}]", ' ', text)
        return re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    
    @staticmethod
    def process_text(text):
        text = VietnameseTextCleaner.remove_html(text)
        text = VietnameseTextCleaner.remove_emoji(text)
        text = VietnameseTextCleaner.remove_url(text)
        text = VietnameseTextCleaner.remove_email(text)
        text = VietnameseTextCleaner.remove_phone_number(text)
        text = VietnameseTextCleaner.remove_hashtags(text)
        return VietnameseTextCleaner.remove_unnecessary_characters(text)


class VietnameseToneNormalizer:
    VOWELS_TABLE = [
        ['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
        ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
        ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
        ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e' ],
        ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
        ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i' ],
        ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o' ],
        ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
        ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
        ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u' ],
        ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
        ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']
    ]
    
    # VOWELS_TO_IDS = {}
    # for i, row in enumerate(VOWELS_TABLE):
    #     for j, char in enumerate(row[:-1]):
    #         VOWELS_TO_IDS[char] = (i, j)
    VOWELS_TO_IDS = {
        'a': (0, 0), 'à': (0, 1), 'á': (0, 2), 'ả': (0, 3), 'ã': (0, 4), 'ạ': (0, 5), 
        'ă': (1, 0), 'ằ': (1, 1), 'ắ': (1, 2), 'ẳ': (1, 3), 'ẵ': (1, 4), 'ặ': (1, 5), 
        'â': (2, 0), 'ầ': (2, 1), 'ấ': (2, 2), 'ẩ': (2, 3), 'ẫ': (2, 4), 'ậ': (2, 5), 
        'e': (3, 0), 'è': (3, 1), 'é': (3, 2), 'ẻ': (3, 3), 'ẽ': (3, 4), 'ẹ': (3, 5), 
        'ê': (4, 0), 'ề': (4, 1), 'ế': (4, 2), 'ể': (4, 3), 'ễ': (4, 4), 'ệ': (4, 5), 
        'i': (5, 0), 'ì': (5, 1), 'í': (5, 2), 'ỉ': (5, 3), 'ĩ': (5, 4), 'ị': (5, 5), 
        'o': (6, 0), 'ò': (6, 1), 'ó': (6, 2), 'ỏ': (6, 3), 'õ': (6, 4), 'ọ': (6, 5), 
        'ô': (7, 0), 'ồ': (7, 1), 'ố': (7, 2), 'ổ': (7, 3), 'ỗ': (7, 4), 'ộ': (7, 5), 
        'ơ': (8, 0), 'ờ': (8, 1), 'ớ': (8, 2), 'ở': (8, 3), 'ỡ': (8, 4), 'ợ': (8, 5), 
        'u': (9, 0), 'ù': (9, 1), 'ú': (9, 2), 'ủ': (9, 3), 'ũ': (9, 4), 'ụ': (9, 5), 
        'ư': (10, 0), 'ừ': (10, 1), 'ứ': (10, 2), 'ử': (10, 3), 'ữ': (10, 4), 'ự': (10, 5), 
        'y': (11, 0), 'ỳ': (11, 1), 'ý': (11, 2), 'ỷ': (11, 3), 'ỹ': (11, 4), 'ỵ': (11, 5)
    }
    
    VINAI_NORMALIZED_TONE = {
        'òa': 'oà', 'Òa': 'Oà', 'ÒA': 'OÀ', 
        'óa': 'oá', 'Óa': 'Oá', 'ÓA': 'OÁ', 
        'ỏa': 'oả', 'Ỏa': 'Oả', 'ỎA': 'OẢ',
        'õa': 'oã', 'Õa': 'Oã', 'ÕA': 'OÃ',
        'ọa': 'oạ', 'Ọa': 'Oạ', 'ỌA': 'OẠ',
        'òe': 'oè', 'Òe': 'Oè', 'ÒE': 'OÈ',
        'óe': 'oé', 'Óe': 'Oé', 'ÓE': 'OÉ',
        'ỏe': 'oẻ', 'Ỏe': 'Oẻ', 'ỎE': 'OẺ',
        'õe': 'oẽ', 'Õe': 'Oẽ', 'ÕE': 'OẼ',
        'ọe': 'oẹ', 'Ọe': 'Oẹ', 'ỌE': 'OẸ',
        'ùy': 'uỳ', 'Ùy': 'Uỳ', 'ÙY': 'UỲ',
        'úy': 'uý', 'Úy': 'Uý', 'ÚY': 'UÝ',
        'ủy': 'uỷ', 'Ủy': 'Uỷ', 'ỦY': 'UỶ',
        'ũy': 'uỹ', 'Ũy': 'Uỹ', 'ŨY': 'UỸ',
        'ụy': 'uỵ', 'Ụy': 'Uỵ', 'ỤY': 'UỴ',
    }


    @staticmethod
    def normalize_unicode(text):
        char1252 = r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'
        charutf8 = r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'
        char_map = dict(zip(char1252.split('|'), charutf8.split('|')))
        return re.sub(char1252, lambda x: char_map[x.group()], text.strip())
    
    
    @staticmethod
    def normalize_sentence_typing(text, vinai_normalization=False):
        # https://github.com/VinAIResearch/BARTpho/blob/main/VietnameseToneNormalization.md
        if vinai_normalization: # Just simply replace the wrong tone with the correct one defined by VinAI
            for wrong, correct in VietnameseToneNormalizer.VINAI_NORMALIZED_TONE.items():
                text = text.replace(wrong, correct)
            return text.strip()
        
        # Or you can use this algorithm developed by Behitek to normalize Vietnamese typing in a sentence 
        words = text.strip().split()
        for index, word in enumerate(words):
            cw = re.sub(r'(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)', r'\1/\2/\3', word).split('/')
            if len(cw) == 3: cw[1] = VietnameseToneNormalizer.normalize_word_typing(cw[1])
            words[index] = ''.join(cw)
        return ' '.join(words)
    
     
    @staticmethod
    def normalize_word_typing(word):
        if not VietnameseToneNormalizer.is_valid_vietnamese_word(word): return word
        chars, vowel_indexes = list(word), []
        qu_or_gi, tonal_mark = False, 0
        
        for index, char in enumerate(chars):
            if char not in VietnameseToneNormalizer.VOWELS_TO_IDS: continue
            row, col = VietnameseToneNormalizer.VOWELS_TO_IDS[char]
            if index > 0 and (row, chars[index - 1]) in [(9, 'q'), (5, 'g')]:
                chars[index] = VietnameseToneNormalizer.VOWELS_TABLE[row][0]
                qu_or_gi = True
                
            if not qu_or_gi or index != 1: vowel_indexes.append(index)
            if col != 0:
                tonal_mark = col
                chars[index] = VietnameseToneNormalizer.VOWELS_TABLE[row][0]
                
        if len(vowel_indexes) < 2:
            if qu_or_gi:
                index = 1 if len(chars) == 2 else 2
                if chars[index] in VietnameseToneNormalizer.VOWELS_TO_IDS:
                    row, _ = VietnameseToneNormalizer.VOWELS_TO_IDS[chars[index]]
                    chars[index] = VietnameseToneNormalizer.VOWELS_TABLE[row][tonal_mark]
                else: chars[1] = VietnameseToneNormalizer.VOWELS_TABLE[5 if chars[1] == 'i' else 9][tonal_mark]
                return ''.join(chars)
            return word
        
        for index in vowel_indexes:
            row, _ = VietnameseToneNormalizer.VOWELS_TO_IDS[chars[index]]
            if row in [4, 8]: # ê, ơ
                chars[index] = VietnameseToneNormalizer.VOWELS_TABLE[row][tonal_mark]
                return ''.join(chars)
            
        index = vowel_indexes[0 if len(vowel_indexes) == 2 and vowel_indexes[-1] == len(chars) - 1 else 1] 
        row, _ = VietnameseToneNormalizer.VOWELS_TO_IDS[chars[index]]
        chars[index] = VietnameseToneNormalizer.VOWELS_TABLE[row][tonal_mark]
        return ''.join(chars)
    
    
    @staticmethod
    def is_valid_vietnamese_word(word):
        vowel_indexes = -1 
        for index, char in enumerate(word):
            if char not in VietnameseToneNormalizer.VOWELS_TO_IDS: continue
            if vowel_indexes in [-1, index - 1]: vowel_indexes = index
            else: return False
        return True
    

class VietnameseTextPreprocessor:
    def __init__(self, vncorenlp_dir='./VnCoreNLP', extra_teencodes=None, max_correction_length=512):
        self.vncorenlp_dir = vncorenlp_dir
        self.extra_teencodes = extra_teencodes
        self._load_vncorenlp()
        self._build_teencodes()
        
        self.max_correction_length = max_correction_length
        self.corrector = pipeline(
            'text2text-generation', model='bmd1905/vietnamese-correction-v2', 
            torch_dtype='bfloat16', device_map='auto', num_workers=os.cpu_count()
        )
        print('bmd1905/vietnamese-correction-v2 is loaded successfully.')
        
    
    def _load_vncorenlp(self):
        self.word_segmenter = None
        if self._get_vncorenlp_files('/VnCoreNLP-1.2.jar') and \
           self._get_vncorenlp_files('/models/wordsegmenter/vi-vocab') and \
           self._get_vncorenlp_files('/models/wordsegmenter/wordsegmenter.rdr'):
            self.word_segmenter = VnCoreNLP(self.vncorenlp_dir + '/VnCoreNLP-1.2.jar', annotators='wseg', quiet=False)
            print('VnCoreNLP word segmenter is loaded successfully.')
        else: print('Failed to load VnCoreNLP word segmenter.')
            

    def _get_vncorenlp_files(self, url_slash):
        local_path = self.vncorenlp_dir + url_slash
        if os.path.exists(local_path): return True
        
        # Check if the folder contains the local_path exists, if not, create it.
        if not os.path.exists(os.path.dirname(local_path)):
            os.makedirs(os.path.dirname(local_path))
        
        download_url = 'https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master' + url_slash
        try: 
            print(f'Downloading {download_url} to {local_path}')
            return urllib.request.urlretrieve(download_url, local_path)
        except urllib.error.HTTPError as e:
            print(f'Failed to download {download_url} due to {e}')
            return False
                
        
    def _build_teencodes(self):
        self.teencodes = {
            'ok': ['okie', 'okey', 'ôkê', 'oki', 'oke', 'okay', 'okê'], 
            'không': ['kg', 'not', 'k', 'kh', 'kô', 'hok', 'ko', 'khong'], 'không phải': ['kp'], 
            'cảm ơn': ['tks', 'thks', 'thanks', 'ths', 'thank'], 'hồi đó': ['hùi đó'], 'muốn': ['mún'],
            
            'rất tốt': ['perfect', '❤️', '😍'], 'dễ thương': ['cute'], 'yêu': ['iu'], 'thích': ['thik'], 
            'tốt': [
                'gud', 'good', 'gút', 'tot', 'nice',
                'hehe', 'hihi', 'haha', 'hjhj', 'thick', '^_^', ':)', '=)'
                '👍', '🎉', '😀', '😂', '🤗', '😙', '🙂'
            ], 
            'bình thường': ['bt', 'bthg'], 'hàg': ['hàng'], 
            'không tốt':  ['lol', 'cc', 'huhu', ':(', '😔', '😓'],
            'tệ': ['sad', 'por', 'poor', 'bad'], 'giả mạo': ['fake'], 
            
            'quá': ['wa', 'wá', 'qá'], 'được': ['đx', 'dk', 'dc', 'đk', 'đc'], 
            'với': ['vs'], 'gì': ['j'], 'rồi': ['r'], 'mình': ['m', 'mik'], 
            'thời gian': ['time'], 'giờ': ['h'], 
        }
        if self.extra_teencodes: 
            for key, values in self.extra_teencodes.items():
                if any(len(value.split()) > 1 for value in values):
                    raise ValueError('The values for each key in extra_teencodes must be single words.')
                self.teencodes.setdefault(key, []).extend(values)
                
        self.teencodes = {word: key for key, values in self.teencodes.items() for word in values}
        teencode_url = 'https://gist.githubusercontent.com/behitek/7d9441c10b3c2739499fc5a4d9ea06fb/raw/df939245b3e841b62af115be4dcb3516dadc9fc5/teencode.txt'
        response = requests.get(teencode_url)
        
        if response.status_code == 200:
            text_data = StringIO(response.text)
            for pair in text_data:
                teencode, true_text = pair.split('\t')
                self.teencodes[teencode.strip()] = true_text.strip()
            self.teencodes = {k: self.teencodes[k] for k in sorted(self.teencodes)}
        else: print('Failed to fetch teencode.txt from', teencode_url)

    
    def normalize_teencodes(self, text):
        words = []
        for word in text.split():
            words.append(self.teencodes.get(word, word))
        return ' '.join(words)
    
    
    def correct_vietnamese_errors(self, texts):
        # https://huggingface.co/bmd1905/vietnamese-correction-v2
        predictions = self.corrector(texts, max_length=self.max_correction_length, truncation=True)
        return [prediction['generated_text'] for prediction in predictions]
        
    
    def word_segment(self, text):
        if self.word_segmenter: 
            words = self.word_segmenter.tokenize(text)
            return ' '.join(sum(words, [])) # Flatten the list of words
        print('There is no VnCoreNLP word segmenter loaded. Please check the VnCoreNLP jar file.')
        return text
        
    
    def process_text(self, text, normalize_tone=True, segment=True):
        text = text.lower()
        if normalize_tone:
            text = VietnameseToneNormalizer.normalize_unicode(text)
            text = VietnameseToneNormalizer.normalize_sentence_typing(text)
        text = VietnameseTextCleaner.process_text(text)
        text = self.normalize_teencodes(text)
        return self.word_segment(text) if segment else text
    
    
    def process_batch(self, texts, correct_errors=True):
        if correct_errors:
            texts = [self.process_text(text, normalize_tone=True, segment=False) for text in texts]
            texts = self.correct_vietnamese_errors(texts)
            return [self.process_text(text, normalize_tone=False, segment=True) for text in texts]
        return [self.process_text(text, normalize_tone=True, segment=True) for text in texts]
    
    
    def close_vncorenlp(self):
        if self.word_segmenter: 
            print('Closing VnCoreNLP word segmenter...')
            self.word_segmenter.close()
    
    
if __name__ == '__main__':
    # You should be careful when using single word replacement for teencodes, because it can cause misinterpretation. 
    # For example, 'giá': ['price', 'gia'] can replace the word 'gia' in 'gia đình', making it become 'giá đình'.
    extra_teencodes = { 
        'khách sạn': ['ks'], 'nhà hàng': ['nhahang'], 'nhân viên': ['nv'],
        'cửa hàng': ['store', 'sop', 'shopE', 'shop'], 
        'sản phẩm': ['sp', 'product'], 'hàng': ['hàg'],
        'giao hàng': ['ship', 'delivery', 'síp'], 'đặt hàng': ['order'], 
        'chuẩn chính hãng': ['authentic', 'aut', 'auth'], 'hạn sử dụng': ['date', 'hsd'],
        'điện thoại': ['dt'],  'facebook': ['fb', 'face'],  
        'nhắn tin': ['nt', 'ib'], 'trả lời': ['tl', 'trl', 'rep'], 
        'feedback': ['fback', 'fedback'], 'sử dụng': ['sd'], 'xài': ['sài'], 
    }
    
    preprocessor = VietnameseTextPreprocessor(vncorenlp_dir='./VnCoreNLP', extra_teencodes=extra_teencodes, max_correction_length=512)
    sample_texts = [
        'Ga giường không sạch, nhân viên quên dọn phòng một ngày. Chất lựơng "ko" đc thỏai mái 😔',
        'Cám ơn Chudu24 rất nhiềuGia đình tôi có 1 kỳ nghỉ vui vẻ.Resort Bình Minh nằm ở vị trí rất đẹp, theo đúng tiêu chuẩn, còn về ăn sáng thì wa dở, chỉ có 2,3 món để chọn',
        'Giá cả hợp líĂn uống thoả thíchGiữ xe miễn phíKhông gian bờ kè thoáng mát Có phòng máy lạnhMỗi tội lúc quán đông thì đợi hơi lâu',
        'May lần trước ăn mì k hà, hôm nay ăn thử bún bắp bò. Có chả tôm viên ăn lạ lạ. Tôm thì k nhiều, nhưng vẫn có tôm thật ở nhân bên trong. ',
        'Ngồi ăn Cơm nhà *tiền thân là quán Bão* Phần vậy là 59k nha. Trưa từ 10h-14h, chiều từ 16h-19h. À,có sữa hạt sen ngon lắmm. #food #foodpic #foodporn #foodholic #yummy #deliciuous'
    ]
    preprocessed_texts = preprocessor.process_batch(sample_texts, correct_errors=True)
    preprocessor.close_vncorenlp()
    print(preprocessed_texts)