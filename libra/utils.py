import os

import tempfile

import logging

TMP_FILES = set()


def get_temp_file(file_name):
    tmp_file = tempfile.NamedTemporaryFile(mode='wb', suffix=file_name, delete=False)

    logging.debug('Created temp file at {} with ext {}'.format(tmp_file.name,
                                                               file_name))

    TMP_FILES.add(tmp_file.name)
    return tmp_file


def delete_temp_file(file_name):
    if file_name and file_name in TMP_FILES:
        logging.debug('Deleting temp file {}'.format(file_name))
        os.remove(file_name)
        TMP_FILES.remove(file_name)
