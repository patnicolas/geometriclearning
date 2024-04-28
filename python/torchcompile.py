

from typing import Callable, List, AnyStr
import torch
from util.timeit import timeit

class TorchCompile(object):
    def __init__(self,
                 map_func: Callable[[torch.tensor], torch.tensor],
                 reduce_func: Callable[[torch.tensor], float]):
        self.map_func = map_func
        self.reduce_func = reduce_func
        self.map_compiled_func = torch.compile(map_func)
        self.reduce_compiled_func = torch.compile(reduce_func)

    def __call__(self, values: List[float]) -> float:
        torch_values = torch.tensor(values)
        raw_output = self.__raw(torch_values, 'raw')
        compiled_output = self.__compiled(torch_values, 'compiled')
        return raw_output - compiled_output

    @timeit
    def __raw(self, torch_values: torch.tensor, label: AnyStr) -> float:
        return self.reduce_func(self.map_func(torch_values))

    @timeit
    def __compiled(self, torch_values: torch.tensor, label: AnyStr) -> float:
        return self.reduce_compiled_func(self.map_compiled_func(torch_values))


if __name__ == '__main__':
    print(torch.backends.mps.is_available())  # the MacOS is higher than 12.3+
    print(torch.backends.mps.is_built())  # MPS is activated

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    import numpy

    def mapper(x: torch.tensor) -> torch.tensor:
        return torch.exp(-x) + torch.sin(x*x)

    def reducer(x: torch.tensor) -> float:
        return torch.sqrt(torch.sum(x)).item()

    torch_compile = TorchCompile(mapper, reducer)
    data_size = 100000
    x = list(range(0, data_size))
    torch_compile(x)
