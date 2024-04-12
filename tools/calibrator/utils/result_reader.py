import struct
from typing import Generator
from utils.position_result import PositionFullResult, PositionTopResult, PositionResult
import utils.cpp_constants as cpp_constants
from enum import Enum
import numpy as np

class ResultReaderType(Enum):
    FULL = 1
    TOP = 2

class ResultReader:
    def __init__(self, path: str, little_endian: bool = True):
        reader_type = None
        try:
            reader_type = path.split('.')[-2]
        except:
            raise ValueError('Could not determine the type of the file from the file_path: {file_path}.'
                             'The file_path should have the format: `output.[output_writer_type].extension`')
        
        if reader_type == 'full':
            self.reader_type = ResultReaderType.FULL
        elif reader_type == 'top':
            self.reader_type = ResultReaderType.TOP
        else:
            raise ValueError(f'Unknown reader_type: {reader_type}')

        self.path = path
        self.little_endian = little_endian

    def read(self) -> Generator[PositionResult, any, any]:
        if self.reader_type == ResultReaderType.FULL:
            return self._read_full_result()
        elif self.reader_type == ResultReaderType.TOP:
            return self._read_top_results()
        

    def _read_full_result(self) -> Generator[PositionFullResult, any, any]:
        le_prefix = '<' if self.little_endian else ''

        with open(self.path, 'rb') as file:
            while True:
                # 1. Read the number of tokens
                bytes = file.read(cpp_constants.UINT_16T_SIZE)
                if not bytes:
                    break
                num_tokens = struct.unpack(le_prefix+'h', bytes)[0]

                # 2. Read the correct token
                bytes = file.read(cpp_constants.UINT_16T_SIZE)
                correct_token = struct.unpack(le_prefix+'h', bytes)[0]

                # 3. Read the data
                token_data = []
                for i in range(num_tokens):
                    bytes = file.read(cpp_constants.FLOAT_SIZE)
                    token_data_item = np.float32(struct.unpack(le_prefix+'f', bytes)[0])
                    token_data.append(token_data_item)
                
                np_token_data = np.array(token_data, dtype=np.float32)
                position_result = PositionFullResult(np_token_data, correct_token)

                # (4. Read and compare the checksum)
                bytes = file.read(cpp_constants.UINT_32T_SIZE)
                checksum = struct.unpack(le_prefix+'I', bytes)[0]
                if checksum != position_result.get_checksum(self.little_endian):
                    raise ValueError(f'Checksum does not match for position_result: {position_result}')

                yield position_result
        file.close()

    
    def _read_top_results(self) -> Generator[PositionTopResult, any, any]:
        le_prefix = '<' if self.little_endian else ''

        with open(self.path, 'rb') as file:
            while True:
                # 1. Read the number of tokens
                bytes = file.read(cpp_constants.UINT_16T_SIZE)
                if not bytes:
                    break
                num_tokens = struct.unpack(le_prefix+'h', bytes)[0]

                # 2. Read the token data
                token_data = []
                for i in range(num_tokens):
                    bytes = file.read(cpp_constants.FLOAT_SIZE)
                    token_data_item = np.float32(struct.unpack(le_prefix+'f', bytes)[0])
                    token_data.append(token_data_item)
                
                np_token_data = np.array(token_data, dtype=np.float32)
                position_result = PositionTopResult(np_token_data, num_tokens)

                # (3. Read and compare the checksum)
                bytes = file.read(cpp_constants.UINT_32T_SIZE)
                checksum = struct.unpack(le_prefix+'I', bytes)[0]
                if checksum != position_result.get_checksum(self.little_endian):
                    raise ValueError(f'Checksum does not match for position_result: {position_result}')
                
                yield position_result
        file.close()