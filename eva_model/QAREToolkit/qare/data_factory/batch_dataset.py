#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-10
'''

import logging
import types
from tqdm import tqdm
from torch.utils import data
from eva_model.QAREToolkit.qare.data_factory.transform_instance import transform_instance


class _Dataset(data.Dataset):

    def __init__(self,
                 instance_generator,
                 transform,
                 **transform_kwargs):
        logging.info("Loading Dataset.")
        if isinstance(instance_generator, list):
            self._data = instance_generator
        elif isinstance(instance_generator, types.GeneratorType):
            self._data = [instance for instance in tqdm(instance_generator)]
        else:
            raise TypeError
        self._transform = transform
        self._transform_kwargs = transform_kwargs

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        instance = self._data[index]
        if self._transform:
            # question_token_idxs, answer_token_idxs, label = self._transform(
            #     instance = instance,
            #     **self._transform_kwargs)
            transform_data = self._transform(
                instance = instance,
                **self._transform_kwargs)
        else:
            raise TypeError("transform should be defined for BatchDataset !")

        return transform_data


class BatchDataset(object):

    def __init__(self,
                 instance_generator,
                 batch_size,
                 shuffle = False,
                 num_workers = 4,
                 transform = transform_instance,
                 **transform_kwargs):

        self.instance_generator = instance_generator
        if instance_generator is None:
            raise ValueError('Empty instances!')
        self._batch_size = batch_size
        self.transform = transform
        self._transform_kwargs = transform_kwargs
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.building_batch_dataset()

    def building_batch_dataset(self):
        self._Dataset = _Dataset(instance_generator = self.instance_generator,
                                 transform = self.transform,
                                 **self._transform_kwargs)

        self._batch_generator = data.DataLoader(dataset = self._Dataset,
                                                batch_size = self._batch_size,
                                                shuffle = self.shuffle,
                                                num_workers = self.num_workers)

    def __len__(self):
        return len(self._Dataset)

    def next(self):
        next_batch_data = self._batch_generator.__iter__().__next__()
        return next_batch_data

    def get_batch_size(self):
        return self._batch_size

    def get_batch_generator(self):
        return self._batch_generator

