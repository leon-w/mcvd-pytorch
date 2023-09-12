from datetime import datetime

import torch
from colorama import Fore, Style


def red(s):
    return f"{Fore.LIGHTRED_EX}{s}{Style.RESET_ALL}"


def yellow(s):
    return f"{Fore.YELLOW}{s}{Style.RESET_ALL}"


def green(s):
    return f"{Fore.GREEN}{s}{Style.RESET_ALL}"


def dtype_to_str(type):
    if type == torch.float32:
        return "f32"
    if type == torch.float64:
        return "f64"
    if type == torch.int32:
        return "i32"
    if type == torch.int64:
        return "i64"
    if type == torch.bool:
        return "bool"
    if type == torch.uint8:
        return "u8"
    if type == torch.int8:
        return "i8"
    if type == torch.int16:
        return "i16"
    if type == torch.float16:
        return "f16"
    if type == torch.bfloat16:
        return "bf16"
    return repr(type)


class FormatObject:
    def __init__(self, o):
        self.o = o

    def __repr__(self):
        if isinstance(self.o, torch.Tensor):
            if self.o.ndim == 0:
                return red(f"T[x={self.o.detach().cpu().numpy()}, {dtype_to_str(self.o.dtype)}]")
            return red(f"T[{tuple(self.o.shape)}, {dtype_to_str(self.o.dtype)}]")

        if isinstance(self.o, tuple):
            return repr(tuple(FormatObject(x) for x in self.o))

        if isinstance(self.o, list):
            return repr([FormatObject(x) for x in self.o])

        return red(self.o)


def p(*args, **kwargs):
    time = green(f"[{datetime.now().strftime('%H:%M:%S')}]")

    items = []

    for arg in args:
        items.append(FormatObject(arg))

    for k, v in kwargs.items():
        items.append(yellow(f"{k}=") + repr(FormatObject(v)))

    print(time, *items)
