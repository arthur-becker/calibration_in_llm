from abc import ABC, abstractmethod
import binascii
import struct
import cpp_constants

class PositionResult(ABC):
    @abstractmethod
    def get_bytes(self):
        raise NotImplementedError

    def get_checksum(self, little_endian=False):
        bytes = self.get_bytes(little_endian)
        return binascii.crc32(bytes)
    
class PositionFullResult(PositionResult):
    def __init__(self, token_data, correct_token):
        self.token_data = token_data
        self.correct_token = correct_token

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
    

    
