from abc import ABC, abstractmethod
import binascii
import struct
import utils.cpp_constants as cpp_constants
import os

class PositionResult(ABC):
    @abstractmethod
    def get_bytes(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_correct_token(self):
        raise NotImplementedError
    
    def get_token_data(self):
        return self.token_data

    def get_checksum(self, little_endian=False):
        bytes = self.get_bytes(little_endian)
        return binascii.crc32(bytes)
    
    def save_position_result(self, name, batch_size=50):
        print(f'Saving position result to file: position_result.{name}.txt')

        # Remove the file if it already exists
        try:
            os.remove(f'position_result.{name}.txt')
        except FileNotFoundError:
            pass

        token_data = self.get_token_data()

        # Save the data to a file
        with open(f'position_result.{name}.txt', 'w') as file:
            file.write(f'Correct token: {self.get_correct_token()}\n')
            file.write(f'len(token_data): {len(token_data)}\n')

            for i in range(0, len(token_data), batch_size):
                file.write(f'token_data[{i}:{i+batch_size}]: {token_data[i:i+batch_size]}\n')
        print('Position result has been saved.')
    
class PositionFullResult(PositionResult):
    def __init__(self, token_data, correct_token):
        self.token_data = token_data
        self.correct_token = correct_token

    def get_correct_token(self):
        return self.correct_token

    def get_bytes(self, little_endian=False):
        prefix = '<' if little_endian else ''
        byteorder = 'little' if little_endian else 'big'

        bytes = bytearray()

        # 1. Save the number of tokens
        num_tokens = len(self.token_data)
        bytes.extend(num_tokens.to_bytes(cpp_constants.UINT_16T_SIZE, byteorder))

        # 2. Save the correct token
        bytes.extend(self.correct_token.to_bytes(cpp_constants.UINT_16T_SIZE, byteorder))

        # 3. Save the token data
        for token in self.token_data:
            bytes.extend(struct.pack(prefix+'f', token))
        return bytes

class PositionTopResult(PositionResult):
    """
    The data for the correct token is saved in `self.token_data[0]`
    """

    def __init__(self, token_data, n):
        """
        token_data: list of floats. The first element is the data for the correct token.
        n: int. The number of tokens in token_data.
        """
        if len(token_data) != n:
            raise ValueError(f'len(token_data) != n: {len(token_data)} != {n}')

        self.token_data = token_data
        self.n = n

    def get_correct_token(self):
        return 0

    def get_bytes(self, little_endian=False):
        prefix = '<' if little_endian else ''
        byteorder = 'little' if little_endian else 'big'

        bytes = bytearray()

        # 1. Save the number of tokens
        bytes.extend(self.n.to_bytes(cpp_constants.UINT_16T_SIZE, byteorder))

        # 2. Save the token data
        for token in self.token_data:
            bytes.extend(struct.pack(prefix+'f', token))
        return bytes
    

    
