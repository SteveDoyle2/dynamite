import os
from typing import TextIO, BinaryIO, Optional
from struct import unpack
import numpy as np


class BinaryReader:
    def __init__(self):
        self.f = None
        self.n = 0
        self.endian = b'<'
        self.uendian = '<'

    def _show(self, op4: BinaryIO, n: int, types: str='ifs', endian: Optional[str]=None):
        """Shows binary data"""
        assert self.n == op4.tell()
        nints = n // 4
        data_bytes = self.f.read(4 * nints)
        strings, ints, floats = self._show_data(data_bytes, types=types, endian=endian)
        self.f.seek(self.n)
        return strings, ints, floats

    def _show_data(self, data_bytes: bytes, types: str='ifs', endian: Optional[str]=None):
        """
        Shows a data block as various types

        Parameters
        ----------
        data : bytes
            the binary string bytes
        types : str; default='ifs'
            i - int
            f - float
            s - string
            d - double (float; 8 bytes)

            l - long (int; 4 bytes)
            q - long long (int; int; 8 bytes)
            I - unsigned int (int; 4 bytes)
            L - unsigned long (int; 4 bytes)
            Q - unsigned long long (int; 8 bytes)
        endian : str; default=None -> auto determined somewhere else in the code
            the big/little endian {>, <}

        .. warning:: 's' is apparently not Python 3 friendly

        """
        if endian is None:
            endian = self._endian
        return _write_data(sys.stdout, data_bytes, types=types, endian=endian)

    def _show_ndata(self, f: BinaryIO, n: int, types: str='ifs') -> None:
        #endian = self._endian
        return self._write_ndata(sys.stdout, f, n, types=types)

    def _write_ndata(self, fout: TextIO, f: BinaryIO, n: int, types: str='ifs') -> None:
        """Useful function for seeing what's going on locally when debugging."""
        endian = self._endian
        assert endian is not None, endian
        nold = self.n
        data_bytes = f.read(n)
        self.n = nold
        f.seek(self.n)
        return _write_data(fout, data_bytes, endian=endian, types=types)

def show_binary_file(file_obj: BinaryIO, n: int, types='ifs') -> str:
    i0 = file_obj.tell()
    data = file_obj.read(n)
    file_obj.seek(i0)
    return show_binary_data(data, types=types)

def show_binary_data(data: bytes, types: str='ifs') -> str:
    assert len(data), data

    ndata = len(data)
    ndata4 = (ndata // 4) * 4
    #ndata8 = (ndata // 8) * 8
    msg = ''
    if 'i' in types:
        ints = np.frombuffer(data[:ndata4], dtype='int32').tolist()
        msg += f'ints: {ints}\n'
    if 'f' in types:
        floats = np.frombuffer(data[:ndata4], dtype='float32').tolist()
        msg += f'floats: {floats}\n'
    if 's' in types:
        msg += f'strings: {str(data)}\n'
    print(msg)
    return msg

#class SecReader(BinaryReader):
    #def __init__(self):
        #super().__init__()

def read_sec(sec_filename: str):
    assert os.path.exists(sec_filename), sec_filename
    with open(sec_filename, 'rb') as f:
        show_binary_file(f, 4*158, types='isf')

        datai = f.read(4*158)
        show_binary_data(f, datai, types='isf')
        x = 1
        data_bytes = f.read()
        data = np.frombuffer(data, dtype='float32')
        dt = data[0]
        data = data[1:]
    time = np.arange(len(data)) * dt
    return time, data

