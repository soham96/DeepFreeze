from bitstring import BitArray
import math
import random

def replace_bits(stream, pos_arr, bs):
    for pos in pos_arr:
        stream.overwrite(bs, pos)

    return stream

stream=BitArray(filename='simple_ffnn/simple_ffnn.xml')

truearray=BitArray([True])
falsearray=BitArray([False])

total_one_to_zero=math.ceil(0.01*len(stream))
total_zero_to_one=math.ceil(0.001*len(stream))

one_indices = [i for i, v in enumerate(stream.bin) if v=='1']
zero_indices = [i for i, v in enumerate(stream.bin) if v=='0']

one_indices=random.sample(one_indices, total_one_to_zero)
zero_indices=random.sample(zero_indices, total_zero_to_one)

stream=replace_bits(stream, one_indices, falsearray)
stream=replace_bits(stream, zero_indices, truearray)

with open('new_file.xml', 'wb') as f:
    stream.tofile(f)
