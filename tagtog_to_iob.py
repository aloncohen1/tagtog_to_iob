import zipfile
import json
from collections import defaultdict
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import spacy

nlp = spacy.load('en_core_web_sm')


class TagTogParser(object):

    def __init__(self, tagtog_zip_path) -> None:
        self.tagtog_zip_path = tagtog_zip_path
        self.tagtog_archive = zipfile.ZipFile(tagtog_zip_path, 'r')
        self.ann_dict = None
        self.ann_leged_df = None
        self.tagtog_df = None
        self.tags_counts = None
        self.iob_tags_df = None

    def get_tagtog_df(self, keep_soup=False):

        print('generating tagtog_df...')

        soups_list = [{'file_name': i.split('pool/')[1].split('.plain')[0],
                       'soup': BeautifulSoup(self.tagtog_archive.read(i))}
                      for i in self.tagtog_archive.namelist() if i.endswith('.html')]  # text html as soup bject

        ann_list = [{'file_name': i.split('pool/')[1].split('.ann')[0],
                     'ann': json.loads(self.tagtog_archive.read(i))}
                    for i in self.tagtog_archive.namelist()[:-1] if i.endswith('.json')]  # annotation dict

        self.ann_dict = json.loads(
            self.tagtog_archive.read(self.tagtog_archive.namelist()[-1]))  # read annotations-legend.json

        self.ann_leged_df = pd.DataFrame(self.ann_dict.items(), columns=['tag_code', 'tag_meaning'])
        self.tagtog_df = pd.DataFrame(soups_list).merge(pd.DataFrame(ann_list),
                                                        on='file_name')  # join html files and text (based on file name)
        self.tagtog_df['text'] = self.tagtog_df['soup'].apply(
            lambda x: x.find_all(attrs={"id": "s1s2v1"})[0].text)  # extact text from html
        self.tagtog_df['text'] = self.tagtog_df['text'].str.encode('ascii', 'ignore').str.decode('ascii').replace(
            r'\s+', ' ', regex=True) \
            .str.strip()  # remove non-standatd characters from text

        if not keep_soup:
            self.tagtog_df.drop('soup', axis=1, inplace=True)

        return self.tagtog_df

    def get_tags_counts(self, enrich_df=False):

        def get_tags(self, ann):
            return {self.ann_dict[i['classId']]: True for i in ann['entities']}

        if self.tagtog_df is None:
            print('TagTogParser has no tagtog_df. generating it...')
            self.get_tagtog_df()

        print('calculating tags counts...')

        if self.tags_counts is None:
            self.tags_counts = pd.DataFrame(self.tagtog_df['ann'].apply(lambda x: get_tags(self, x)).to_list()).fillna(
                False)

        if enrich_df:
            self.tagtog_df = self.tagtog_df.join(self.tags_counts)

        return self.tags_counts.sum()

    def get_tags_text(self, enrich_df=False):

        def extract_tags_text(row):
            entities_text_dict = defaultdict(list)
            for tag in row['entities']:
                entities_text_dict[self.ann_dict[tag['classId']] + '_text'].append(tag['offsets'][0]['text'])
            return entities_text_dict

        tags_text_df = pd.DataFrame(self.tagtog_df['ann'].apply(lambda x: extract_tags_text(x)).to_list())

        if enrich_df:
            self.tagtog_df = self.tagtog_df.join(tags_text_df)

        return tags_text_df

    def plot_tags_count(self):

        if self.tags_counts is None:
            print('TagTogParser has no tags_counts. generating it...')
            self.get_tags_counts()

        self.tags_counts.sum().sort_values(ascending=False).plot(kind='bar', ylabel='n_tags', title='Tags Distribution')

    def convert_tagging_to_iob(self, row):

        def parse_tag_name(ann_entity):
            tag_name = self.ann_dict[ann_entity['classId']]
            parsed_tags = []
            for index, word in enumerate(ann_entity['offsets'][0]['text'].split()):
                if index == 0:
                    parsed_tags.append(f'B-{tag_name}')
                else:
                    parsed_tags.append(f'I-{tag_name}')

            return parsed_tags

        text, ann = row['text'], row['ann']

        if row.name % 100 == 0 and row.name != 0:
            print(f'{row.name} instances processed')

        all_text_tags = []
        text = [token.text for token in nlp(text)]
        text_copy = text.copy()
        for ann_entity in ann['entities']:
            tags_name = parse_tag_name(ann_entity)
            all_text_tags.extend(tags_name)
            tagged_text = [token.text for token in nlp(ann_entity['offsets'][0]['text'])]

            tag_index = np.array(list(text[idx: idx + len(tagged_text)] == tagged_text
                                      for idx in range(len(text) - len(tagged_text) + 1))).argmax()

            for index, tag in enumerate(tags_name):
                text[tag_index + index] = tag

        return text_copy, ['O' if i not in all_text_tags else i for i in text]

    def get_iob_tags_df(self, enrich_df=False):

        if self.tagtog_df is None:
            print('TagTogParser has no tagtog_df. generating it...')
            self.get_tagtog_df()

        print('converting tags to IOB format...')

        self.iob_tags_df = self.tagtog_df.apply(lambda x: self.convert_tagging_to_iob(x), axis=1, result_type='expand') \
            .rename(columns={0: 'text_list', 1: 'tags_list'})

        if enrich_df:
            self.tagtog_df = pd.concat([self.tagtog_df, self.iob_tags_df], axis=1)

        print('get_iob_tags_df - Done!')

        return self.iob_tags_df
