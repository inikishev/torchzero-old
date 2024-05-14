import math
import torch
def _map_to_base(number:int, base):
    """
    Convert an integer into a list of digits of that integer in a given base.

    Args:
        number (int): The integer to convert.
        base (int): The base to convert the integer to.

    Returns:
        numpy.ndarray: An array of digits representing the input integer in the given base.
    """
    if number == 0: return torch.tensor([0])
    # Convert the input numbers to their digit representation in the given base
    digits = torch.tensor([number])
    base_digits = (digits // base**(torch.arange(int(math.log(number) / math.log(base)), -1, -1))) % base

    return base_digits

def _extend_to_length(tensor:torch.Tensor, length:int):
    """Extend vector to given size by concatenating zeros to the left"""
    if tensor.size(0) == length: return tensor
    else: return torch.cat((torch.zeros(length-tensor.size(0)), tensor))

class _GridSearch:
    """Utility class, maps integers from 0 to num_combinations to every possible parameter combination"""
    def __init__(self, num, step, domain):
        self.num = num
        self.step = step
        self.domain = domain
        self.domain_normalized = [int(i/step) for i in domain] # we get 3, 4, 5
        self.add_term = self.domain_normalized[0]
        self.domain_from0 = [i - self.add_term for i in self.domain_normalized] # we get 0, 1, 2
        self.base = self.domain_from0[1] + 1

        self.num_combinations = self.base ** num
        self.cur = 0

    def get(self, v): return (_extend_to_length(_map_to_base(v, self.base), self.num) + self.add_term) * self.step
    
def _iproduct2(iterable1, iterable2):
    """Source: https://github.com/sympy/sympy/blob/master/sympy/utilities/iterables.py"""
    it1 = iter(iterable1)
    it2 = iter(iterable2)
    elems1 = []
    elems2 = []
    sentinel = object()
    def append(it, elems):
        e = next(it, sentinel)
        if e is not sentinel:
            elems.append(e)
    n = 0
    append(it1, elems1)
    append(it2, elems2)
    while n <= len(elems1) + len(elems2):
        for m in range(n-len(elems1)+1, len(elems2)):
            yield (elems1[n-m], elems2[m])
        n += 1
        append(it1, elems1)
        append(it2, elems2)

def iproduct(*iterables):
    """Source: https://github.com/sympy/sympy/blob/master/sympy/utilities/iterables.py"""
    if len(iterables) == 0:
        yield ()
        return
    elif len(iterables) == 1:
        for e in iterables[0]:
            yield (e,)
    elif len(iterables) == 2:
        yield from _iproduct2(*iterables)
    else:
        first, others = iterables[0], iterables[1:]
        for ef, eo in _iproduct2(first, iproduct(*others)):
            yield (ef,) + eo