# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np


class DataSetFile(object):
    def __init__(self, data_path):
        """
        Descriptions:
            Construct a DataSet.it is used for classification task, if for regression, it is necessary to
            do minor modifying.
        Args:
            data_path： data folder, it contains multiple subdirs named by each class name.
        """
        self.images_path_list, self.labels_list = self._read_fileslist(data_path)
        assert len(self.images_path_list) == len(self.labels_list)
        self._num_examples = len(self.images_path_list)
        self._data_index = np.arange(self._num_examples)
        self._epochs_completed = 0
        self._index_in_epoch = 0

        self.flag = 0

        self.names_dict = {}

    def _read_fileslist(self, data_path):
        images_path_list, labels_list = [], []
        self.names_dict = {}
        for index, name in enumerate(os.listdir(data_path)):
            self.names_dict[name] = index
            class_path = os.path.join(data_path, name)
            for img in os.listdir(class_path):
                img_path = os.path.join(class_path, img)
                images_path_list.append(img_path)
                labels_list.append(index)
        return images_path_list, labels_list

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=False):
        """
        Return the next `batch_size` examples from this data set.

        """

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._data_index = self._data_index[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            _data_index_rest_part = self._data_index[start:self._num_examples]
            imgs_batch_rest, labels_batch_rest = self._read_batch_data(_data_index_rest_part)
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._data_index = self._data_index[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            _data_index_new_part = self._data_index[start:end]
            imgs_batch_new_part, labels_batch_new_part = self._read_batch_data(_data_index_new_part)
            imgs_batch = np.concatenate((imgs_batch_rest, imgs_batch_new_part), axis=0)
            labels_batch = np.concatenate((labels_batch_rest, labels_batch_new_part), axis=0)
            return imgs_batch, labels_batch
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            imgs_batch, labels_batch = self._read_batch_data(self._data_index[start:end])
            return imgs_batch, labels_batch

    def _read_batch_data(self, batch_imgs_index):
        """
        Args:
            img_index:numpy array with shape [num_samples,]
        """
        batch_size = batch_imgs_index.shape[0]
        batch_imgs = None
        batch_lables = np.zeros(batch_size)

        for i in range(batch_size):
            self.flag += 1
            if i == 0:
                filename = self.images_path_list[batch_imgs_index[i]]
                img = cv2.imread(filename=filename)
                cv2.imshow(str(self.flag) + str(self.labels_list[batch_imgs_index[i]]), img)
                cv2.waitKey()
                img_size = img.shape
                batch_imgs = np.zeros([batch_size, img_size[0], img_size[1], img_size[2]])
                batch_imgs[i, :, :, :] = img
                batch_lables[i] = self.labels_list[batch_imgs_index[i]]
            else:
                filename = self.images_path_list[batch_imgs_index[i]]
                img = cv2.imread(filename=filename)
                cv2.imshow(str(self.flag) + str(self.labels_list[batch_imgs_index[i]]), img)
                cv2.waitKey()
                batch_imgs[i, :, :, :] = img
                batch_lables[i] = self.labels_list[batch_imgs_index[i]]
        return batch_imgs, batch_lables


if __name__ == '__main__':

    data = DataSetFile('/home/pi/stone/Notes/tensorflow/TFRecords/data')
    for i in range(10):
        images_batch, labels_batch = data.next_batch(batch_size=3, shuffle=True)
        print labels_batch
