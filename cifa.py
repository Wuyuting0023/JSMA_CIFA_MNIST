# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tempfile
import warnings

from cleverhans import dataset

utils_CIFAR10_warning = "cleverhans.utils_CIFAR10 is deprecrated and will be " \
                      "removed on or after 2019-03-26. Switch to " \
                      "cleverhans.dataset instead."


def maybe_download_CIFAR10_file(file_name, datadir=None, force=False):
  warnings.warn(utils_CIFAR10_warning)
  url = os.path.join('http://yann.lecun.com/exdb/cifar10/', file_name)
  return dataset.maybe_download_file(url, datadir=None, force=False)


def download_and_parse_CIFAR10_file(file_name, datadir=None, force=False):
  warnings.warn(utils_CIFAR10_warning)
  return dataset.download_and_parse_cifar10_file(file_name, datadir=None,
                                               force=False)


def data_CIFAR10(datadir=tempfile.gettempdir(), train_start=0,
               train_end=60000, test_start=0, test_end=10000):
  warnings.warn(utils_CIFAR10_warning)
  CIFAR10 = dataset.CIFAR10(train_start=train_start,
                        train_end=train_end,
                        test_start=test_start,
                        test_end=test_end,
                        center=False)
  return CIFAR10.get_set('train') + CIFAR10.get_set('test')
