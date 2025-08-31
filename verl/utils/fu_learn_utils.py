# 用于学习，便于打印调试
import os
from typing import Dict
import torch
from verl.protocol import DataProto

def print_values(value_dict:Dict, in_func_name, annotate_str="", has_pid=True, has_rank=True, **kwargs):
    if has_pid:
        pid = os.getpid()
    if has_rank:
        try:
            rank = torch.distributed.get_rank()
        except Exception as e:
            rank = None
    
    out_str = f"[Pid{pid}, rank{rank}] ({in_func_name}) {annotate_str} "

    for name,value in value_dict.items():
        if kwargs.get('value', True): # 输出具体值
            if value is None: 
                out_str += f"{name}:None "
            elif isinstance(value,torch.Tensor):
                out_str += f"{name}:{value} "
            elif isinstance(value,DataProto):
                out_str += f"{name}:{value.batch} "
            elif isinstance(value,list):
                for i,v in enumerate(value):
                    out_str += f"{name}[{i}]:{v} "
            elif isinstance(value,dict):
                for key,val in value.items():
                    out_str += f"{name}.{key}:{val} "
            else:
                out_str += f"{name}:{value} "
        if kwargs.get('bs',True) and not kwargs.get('value'): # 输出bs，并且输出了具体值的话就没必要再输出bs了
            if value is None: 
                out_str += f"{name}:None "
            elif isinstance(value,torch.Tensor):
                out_str += f"{name} bs={value.shape[0]} "
            elif isinstance(value,DataProto):
                out_str += f"{name} bs={value.batch.batch_size} "
            else:
                out_str += f"{name} type={type(value)} bs={len(value)} "

    print(out_str)