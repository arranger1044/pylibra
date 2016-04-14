import sh
import os

LIBRA_BIN_PATH = '/usr/local/bin'
LIBRA_SCRIPT = 'libra'
TMP_DIR = '/tmp'


class LibraConfig(object):

    _bin_path = LIBRA_BIN_PATH
    _script_file = LIBRA_SCRIPT
    _libra_script_path = os.path.join(_bin_path,
                                      _script_file)
    _num_dont_care_val = -1

    @classmethod
    def bin_path(cls):
        return cls._bin_path

    @classmethod
    def script_file(cls):
        return cls._script_file

    @classmethod
    def libra_cmd(cls):
        return sh.Command(cls._libra_script_path)

    @classmethod
    def temp_file_path(cls, file_name):
        return os.path.join(TMP_DIR, file_name)

    @classmethod
    def dont_care_val(cls):
        return cls._num_dont_care_val
