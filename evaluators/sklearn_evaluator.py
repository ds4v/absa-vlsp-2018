import sys
import pandas as pd
from tqdm import tqdm
from IPython.display import display_html
from sklearn.metrics import classification_report
from processors.vlsp2018_processor import PolarityMapping
sys.path.append('..')


class VLSP2018SklearnEvaluator:
    def __init__(self, y_test, y_pred, aspect_category_names): 
        aspect_cate_test, aspect_cate_pred = [], []
        aspect_cate_polar_test, aspect_cate_polar_pred = [], []

        for row_test, row_pred in zip(y_test, y_pred):
            for index, (col_test, col_pred) in enumerate(zip(row_test, row_pred)):
                aspect_cate_test.append(aspect_category_names[index] if col_test != 0 else 'None#None')
                aspect_cate_pred.append(aspect_category_names[index] if col_pred != 0 else 'None#None')
                aspect_cate_polar_test.append(aspect_category_names[index] + f',{PolarityMapping.INDEX_TO_POLARITY[col_test]}')
                aspect_cate_polar_pred.append(aspect_category_names[index] + f',{PolarityMapping.INDEX_TO_POLARITY[col_pred]}')

        self.aspect_cate_polar_report = classification_report(aspect_cate_polar_test, aspect_cate_polar_pred, output_dict=True, zero_division=1.0)
        self.aspect_cate_report = classification_report(aspect_cate_test, aspect_cate_pred, output_dict=True, zero_division=1.0)
        self.polarity_report = classification_report(y_pred.flatten(), y_pred.flatten(), target_names=PolarityMapping.POLARITY_TO_INDEX, output_dict=True)
        self._merge_all_reports()
        self._build_macro_avg_df()
        
    
    def report(self, report_type='all'):
        if report_type.lower() == 'all': self.display_all_reports()
        elif report_type.lower() == 'aspect#category,polarity': return pd.DataFrame(self.aspect_cate_polar_report).T
        elif report_type.lower() == 'aspect#category': return pd.DataFrame(self.aspect_cate_report).T
        elif report_type.lower() == 'polarity': return pd.DataFrame(self.polarity_report).T
        elif report_type.lower() == 'macro_avg': return self.macro_avg_df()
        else: raise ValueError('report_type must be in ["all", "aspect#category,polarity", "aspect#category", "polarity", "macro_avg"]')
        
        
    def _merge_all_reports(self):
        self.merged_report = {}
        for key, metrics in self.aspect_cate_polar_report.items():
            # Check if key in the form of 'aspect#category,polarity' (Check if it's not 'accuracy' or 'macro avg' or 'weighted avg')
            if key in ['accuracy', 'macro avg', 'weighted avg']:
                self.merged_report[key] = {
                    'aspect#category': self.aspect_cate_report[key],
                    'aspect#category,polarity': metrics
                }
            else:
                aspect_cate, polarity = key.split(',')
                if aspect_cate not in self.merged_report:
                    self.merged_report[aspect_cate] = {'aspect#category': self.aspect_cate_report[aspect_cate]}
                self.merged_report[aspect_cate][polarity] = metrics
                
                
    def _build_macro_avg_df(self):
        self.macro_avg_df = pd.DataFrame([{
            'accuracy': f"{report['accuracy']:.3f}", 
            # **{metric: report['macro avg'][metric] for metric in report['macro avg'] if metric != 'accuracy'}
            'precision': f"{report['macro avg']['precision']:.3f}",
            'recall': f"{report['macro avg']['recall']:.3f}",
            'f1-score': f"{report['macro avg']['f1-score']:.3f}",
            'support': report['macro avg']['support']
        } for report in [self.aspect_cate_polar_report, self.aspect_cate_report, self.polarity_report]])
        self.macro_avg_df.index = ['Aspect#Category,Polarity', 'Aspect#Category', 'Polarity']
    
        
    def _display_all_reports(self):
        metric_names = list(self.merged_report.values())[0]['aspect#category']
        html_str = f"""
            <tr>
                <th style="font-weight: bold; text-align: center;" rowspan="2">ACSA Report (w/o "None" polarity)</th>
                <th style="font-weight: bold; text-align: center;" colspan="{len(metric_names)}">Aspect#Category</th>
                <th style="font-weight: bold; text-align: center;" colspan="{len(metric_names)}">Aspect#Category,Polarity</th>
            </tr>
            <tr>
                {''.join([f'<th>{metric_name}</th>' for metric_name in metric_names] * 2)}
            </tr>
        """

        for key, merged_metrics in tqdm(self.merged_report.items()):
            if key in ['accuracy', 'macro avg', 'weighted avg']: continue
            polarities = merged_metrics.keys() - {'aspect#category', 'None'}
            aspect_cate_html = ''.join(
                f'<td rowspan="{len(polarities)}">{value if name == "support" else f"{value:.3f}"}</td>'
                for name, value in self.merged_report[key]['aspect#category'].items()
            )
            for index, polarity in enumerate(polarities):
                aspect_cate_polar_html = ''.join(
                    f'<td>{value if name == "support" else f"{value:.3f}"}</td>'
                    for name, value in self.merged_report[key][polarity].items()
                )
                html_str += f"""
                    <tr>
                        <td>{key},{polarity}</td>
                        {aspect_cate_html if index == 0 else ''}
                        {aspect_cate_polar_html}
                    </tr>
                """

        display_html(f'''
            <div style="display: flex; align-items: flex-start; flex-wrap: nowrap">
                <table style="margin-right: 10px">{html_str}</table> 
                <div style="display: flex; align-items: center; flex-direction: column">
                    <b>Polarity Report</b><br>
                    {pd.DataFrame(self.polarity_report).T.to_html()}<br>
                    <b>Macro Avg Report</b><br>
                    {self.macro_avg_df.to_html()}
                </div>
            </div>
        ''', raw=True)      