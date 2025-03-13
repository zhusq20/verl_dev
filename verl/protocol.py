# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement base data transfer protocol between any two functions, modules.
We can subclass Protocol to define more detailed batch info with specific keys
"""
import pickle
import numpy as np
import pandas as pd
import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Union

import torch
import tensordict
from tensordict import TensorDict
from torch.utils.data import DataLoader, Dataset

from verl.utils.py_functional import union_two_dict

__all__ = ['DataProto', 'union_tensor_dict']

try:
    tensordict.set_lazy_legacy(False).set()
except:
    pass


def pad_dataproto_to_divisor(data: 'DataProto', size_divisor: int):
    """Pad a DataProto to size divisible by size_divisor

    Args:
        size_divisor (int): size divisor

    Returns:
        data: (DataProto): the padded DataProto
        pad_size (int)
    """
    assert isinstance(data, DataProto), 'data must be a DataProto'
    if len(data) % size_divisor != 0:
        pad_size = size_divisor - len(data) % size_divisor
        padding_protos = []
        remaining_pad = pad_size
        while remaining_pad > 0:
            take_size = min(remaining_pad, len(data))
            padding_protos.append(data[:take_size])
            remaining_pad -= take_size
        data_padded = DataProto.concat([data] + padding_protos)
    else:
        pad_size = 0
        data_padded = data
    return data_padded, pad_size


def unpad_dataproto(data: 'DataProto', pad_size):
    if pad_size != 0:
        data = data[:-pad_size]
    return data


def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    """Union two tensordicts."""
    assert tensor_dict1.batch_size == tensor_dict2.batch_size, \
        f'Two tensor dict must have identical batch size. Got {tensor_dict1.batch_size} and {tensor_dict2.batch_size}'
    for key in tensor_dict2.keys():
        if key not in tensor_dict1.keys():
            tensor_dict1[key] = tensor_dict2[key]
        else:
            assert tensor_dict1[key].equal(tensor_dict2[key]), \
                f'{key} in tensor_dict1 and tensor_dict2 are not the same object'

    return tensor_dict1


def union_numpy_dict(tensor_dict1: dict[np.ndarray], tensor_dict2: dict[np.ndarray]) -> dict[np.ndarray]:
    for key, val in tensor_dict2.items():
        if key in tensor_dict1:
            assert isinstance(tensor_dict2[key], np.ndarray)
            assert isinstance(tensor_dict1[key], np.ndarray)
            # to properly deal with nan and object type
            assert pd.DataFrame(tensor_dict2[key]).equals(pd.DataFrame(tensor_dict1[key])), \
                f'{key} in tensor_dict1 and tensor_dict2 are not the same object'
        tensor_dict1[key] = val

    return tensor_dict1


def list_of_dict_to_dict_of_list(list_of_dict: list[dict]):
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()
    output = {key: [] for key in keys}
    for data in list_of_dict:
        for key, item in data.items():
            """hack for now"""
            """如果"""
            # assert key in output
            if key in output:
                output[key].append(item)
    return output


def fold_batch_dim(data: 'DataProto', new_batch_size):
    """
    Fold a batch dim from [bsz, xxx] into [new_bsz, bsz // new_bsz, xxx]
    """
    batch_size = data.batch.batch_size[0]

    assert batch_size % new_batch_size == 0

    tensor: TensorDict = data.batch
    non_tensor = data.non_tensor_batch

    tensor = tensor.view(new_batch_size, -1)
    tensor.auto_batch_size_(batch_dims=1)

    for key, val in non_tensor.items():
        non_tensor[key] = np.reshape(val, newshape=(new_batch_size, -1, *val.shape[1:]))

    return DataProto(batch=tensor, non_tensor_batch=non_tensor, meta_info=data.meta_info)


def unfold_batch_dim(data: 'DataProto', batch_dims=2):
    """
    Unfold the first n dims as new batch dim
    """
    tensor: TensorDict = data.batch
    non_tensor = data.non_tensor_batch
    tensor.auto_batch_size_(batch_dims=batch_dims)
    tensor = tensor.view(-1)

    batch_size = tensor.batch_size[0]

    non_tensor_new = {}

    for key, val in non_tensor.items():
        non_tensor_new[key] = np.reshape(val, newshape=(batch_size, *val.shape[batch_dims:]))

    return DataProto(batch=tensor, non_tensor_batch=non_tensor_new, meta_info=data.meta_info)


def collate_fn(x: list['DataProtoItem']):
    batch = []
    non_tensor_batch = []
    for data in x:
        batch.append(data.batch)
        non_tensor_batch.append(data.non_tensor_batch)
    batch = torch.stack(batch).contiguous()
    non_tensor_batch = list_of_dict_to_dict_of_list(non_tensor_batch)
    for key, val in non_tensor_batch.items():
        non_tensor_batch[key] = np.array(val, dtype=object)
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


@dataclass
class DataProtoItem:
    # TODO(zhangchi.usc1992) add consistency check
    batch: TensorDict = None
    non_tensor_batch: Dict = field(default_factory=dict)
    meta_info: Dict = field(default_factory=dict)


@dataclass
class DataProto:
    """
    A DataProto is a data structure that aims to provide a standard protocol for data exchange between functions.
    It contains a batch (TensorDict) and a meta_info (Dict). The batch is a TensorDict https://pytorch.org/tensordict/.
    TensorDict allows you to manipulate a dictionary of Tensors like a single Tensor. Ideally, the tensors with the
    same batch size should be put inside batch.
    """
    batch: TensorDict = None
    non_tensor_batch: Dict = field(default_factory=dict)
    meta_info: Dict = field(default_factory=dict)

    def __post_init__(self):
        # perform necessary checking
        self.check_consistency()

    def __len__(self):
        if self.batch is not None:
            return self.batch.batch_size[0]
        elif self.non_tensor_batch is not None and len(self.non_tensor_batch) > 0:
            random_key = list(self.non_tensor_batch.keys())[0]
            return self.non_tensor_batch[random_key].shape[0]
        else:
            return 0

    def __getitem__(self, item):
        tensor_data = self.batch[item]
        non_tensor_data = {key: val[item] for key, val in self.non_tensor_batch.items()}
        return DataProtoItem(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=self.meta_info)

    def __getstate__(self):
        import io
        buffer = io.BytesIO()
        if tensordict.__version__ >= '0.5.0' and self.batch is not None:
            self.batch = self.batch.contiguous()
            self.batch = self.batch.consolidate()
        torch.save(self.batch, buffer)
        buffer_bytes = buffer.getvalue()
        return buffer_bytes, self.non_tensor_batch, self.meta_info

    def __setstate__(self, data):
        import io
        batch_deserialized_bytes, non_tensor_batch, meta_info = data
        batch_deserialized = io.BytesIO(initial_bytes=batch_deserialized_bytes)
        batch = torch.load(batch_deserialized,
                           weights_only=False,
                           map_location='cpu' if not torch.cuda.is_available() else None)
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info

    def save_to_disk(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_disk(filepath) -> 'DataProto':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            return data

    def print_size(self, prefix=""):
        size_of_tensordict = 0
        for key, tensor in self.batch.items():
            size_of_tensordict += tensor.element_size() * tensor.numel()
        size_of_numpy_array = 0
        for key, numpy_array in self.non_tensor_batch.items():
            size_of_numpy_array += numpy_array.nbytes

        size_of_numpy_array /= 1024**3
        size_of_tensordict /= 1024**3

        message = f'Size of tensordict: {size_of_tensordict} GB, size of non_tensor_batch: {size_of_numpy_array} GB'

        if prefix:
            message = f'{prefix}, ' + message
        print(message)

    def check_consistency(self):
        """Check the consistency of the DataProto. Mainly for batch and non_tensor_batch
        We expose this function as a public one so that user can call themselves directly
        """
        if self.batch is not None:
            assert len(self.batch.batch_size) == 1, 'only support num_batch_dims=1'

        if self.non_tensor_batch is not None:
            for key, val in self.non_tensor_batch.items():
                assert isinstance(val, np.ndarray)

        if self.batch is not None and len(self.non_tensor_batch) != 0:
            # TODO: we can actually lift this restriction if needed
            assert len(self.batch.batch_size) == 1, 'only support num_batch_dims=1 when non_tensor_batch is not empty.'

            batch_size = self.batch.batch_size[0]
            for key, val in self.non_tensor_batch.items():
                assert isinstance(
                    val, np.ndarray
                ) and val.dtype == object, 'data in the non_tensor_batch must be a numpy.array with dtype=object'
                assert val.shape[0] == batch_size, f'key {key} length {len(val)} is not equal to batch size {batch_size}'
                # if val.shape[0] != batch_size:
                #     self.non_tensor_batch[key] = val[:batch_size]

    @classmethod
    def from_single_dict(cls, data: Dict[str, Union[torch.Tensor, np.ndarray]], meta_info=None):
        tensors = {}
        non_tensors = {}

        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key] = val
            elif isinstance(val, np.ndarray):
                non_tensors[key] = val
            else:
                raise ValueError(f'Unsupported type in data {type(val)}')

        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    @classmethod
    def from_dict(cls, tensors: Dict[str, torch.Tensor], non_tensors=None, meta_info=None, num_batch_dims=1):
        """Create a DataProto from a dict of tensors. This assumes that
        1. All the tensor in tensors have the same dim0
        2. Only dim0 is the batch dim
        """
        assert len(tensors) > 0, 'tensors must not be empty'
        assert num_batch_dims > 0, 'num_batch_dims must be greater than zero'
        if non_tensors is not None:
            assert num_batch_dims == 1, 'only support num_batch_dims=1 when non_tensors is not None.'

        if meta_info is None:
            meta_info = {}
        if non_tensors is None:
            non_tensors = {}

        assert isinstance(non_tensors, dict)

        # get and check batch size
        batch_size = None
        pivot_key = None
        for key, tensor in tensors.items():
            if batch_size is None:
                batch_size = tensor.shape[:num_batch_dims]
                pivot_key = key
            else:
                current_batch = tensor.shape[:num_batch_dims]
                assert batch_size == current_batch, \
                    f'Not all the tensor in tensors have the same batch size with batch_dims={num_batch_dims}. Got {pivot_key} has {batch_size}, {key} has {current_batch}'

        for key, val in non_tensors.items():
            non_tensors[key] = np.array(val, dtype=object)

        tensor_dict = TensorDict(source=tensors, batch_size=batch_size)
        return cls(batch=tensor_dict, non_tensor_batch=non_tensors, meta_info=meta_info)

    def to(self, device) -> 'DataProto':
        """move the batch to device

        Args:
            device (torch.device, str): torch device

        Returns:
            DataProto: the current DataProto

        """
        if self.batch is not None:
            self.batch = self.batch.to(device)
        return self

    def select(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None, deepcopy=False) -> 'DataProto':
        """Select a subset of the DataProto via batch_keys and meta_info_keys

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to select
            meta_info_keys (list, optional): a list of keys indicating the meta info to select

        Returns:
            DataProto: the DataProto with the selected batch_keys and meta_info_keys
        """
        # TODO (zhangchi.usc1992) whether to copy
        if batch_keys is not None:
            batch_keys = tuple(batch_keys)
            sub_batch = self.batch.select(*batch_keys)
        else:
            sub_batch = self.batch

        if non_tensor_batch_keys is not None:
            non_tensor_batch = {key: val for key, val in self.non_tensor_batch.items() if key in non_tensor_batch_keys}
        else:
            non_tensor_batch = self.non_tensor_batch

        if deepcopy:
            non_tensor_batch = copy.deepcopy(non_tensor_batch)

        if meta_info_keys is not None:
            sub_meta_info = {key: val for key, val in self.meta_info.items() if key in meta_info_keys}
        else:
            sub_meta_info = self.meta_info

        if deepcopy:
            sub_meta_info = copy.deepcopy(sub_meta_info)

        return DataProto(batch=sub_batch, non_tensor_batch=non_tensor_batch, meta_info=sub_meta_info)

    def pop(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None) -> 'DataProto':
        """Pop a subset of the DataProto via `batch_keys` and `meta_info_keys`

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to pop
            meta_info_keys (list, optional): a list of keys indicating the meta info to pop

        Returns:
            DataProto: the DataProto with the poped batch_keys and meta_info_keys
        """
        assert batch_keys is not None
        if meta_info_keys is None:
            meta_info_keys = []
        if non_tensor_batch_keys is None:
            non_tensor_batch_keys = []

        tensors = {}
        # tensor batch
        for key in batch_keys:
            assert key in self.batch.keys()
            tensors[key] = self.batch.pop(key)
        non_tensors = {}
        # non tensor batch
        for key in non_tensor_batch_keys:
            assert key in self.non_tensor_batch.keys()
            non_tensors[key] = self.non_tensor_batch.pop(key)
        meta_info = {}
        for key in meta_info_keys:
            assert key in self.meta_info.keys()
            meta_info[key] = self.meta_info.pop(key)
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    def rename(self, old_keys=None, new_keys=None) -> 'DataProto':
        """
        Note that this function only rename the key in the batch
        """

        def validate_input(keys):
            if keys is not None:
                if isinstance(keys, str):
                    keys = [keys]
                elif isinstance(keys, list):
                    pass
                else:
                    raise TypeError(f'keys must be a list or a string, but got {type(keys)}')
            return keys

        old_keys = validate_input(old_keys)
        new_keys = validate_input(new_keys)

        if len(new_keys) != len(old_keys):
            raise ValueError(
                f'new_keys and old_keys must have the same length, but got {len(new_keys)} and {len(old_keys)}')

        self.batch.rename_key_(tuple(old_keys), tuple(new_keys))

        return self

    def union(self, other: 'DataProto') -> 'DataProto':
        """Union with another DataProto. Union batch and meta_info separately.
        Throw an error if

        - there are conflict keys in batch and they are not equal
        - the batch size of two data batch is not the same
        - there are conflict keys in meta_info and they are not the same.

        Args:
            other (DataProto): another DataProto to union

        Returns:
            DataProto: the DataProto after union
        """
        self.batch = union_tensor_dict(self.batch, other.batch)
        self.non_tensor_batch = union_numpy_dict(self.non_tensor_batch, other.non_tensor_batch)
        self.meta_info = union_two_dict(self.meta_info, other.meta_info)
        return self

    def make_iterator(self, mini_batch_size, epochs, seed=None, dataloader_kwargs=None):
        """Make an iterator from the DataProto. This is built upon that TensorDict can be used as a normal Pytorch
        dataset. See https://pytorch.org/tensordict/tutorials/data_fashion for more details.

        Args:
            mini_batch_size (int): mini-batch size when iterating the dataset. We require that
                ``batch.batch_size[0] % mini_batch_size == 0``
            epochs (int): number of epochs when iterating the dataset.
            dataloader_kwargs: internally, it returns a DataLoader over the batch.
                The dataloader_kwargs is the kwargs passed to the DataLoader

        Returns:
            Iterator: an iterator that yields a mini-batch data at a time. The total number of iteration steps is
            ``self.batch.batch_size * epochs // mini_batch_size``
        """
        assert self.batch.batch_size[0] % mini_batch_size == 0, f"{self.batch.batch_size[0]} % {mini_batch_size} != 0"
        # we can directly create a dataloader from TensorDict
        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = None

        assert isinstance(dataloader_kwargs, Dict)
        train_dataloader = DataLoader(dataset=self,
                                      batch_size=mini_batch_size,
                                      collate_fn=collate_fn,
                                      generator=generator,
                                      **dataloader_kwargs)

        def get_data():
            for _ in range(epochs):
                for d in train_dataloader:
                    d.meta_info = self.meta_info
                    yield d

        return iter(get_data())


    def partial_rollout(self, world_size) -> List['DataProto']:
        """Split the batch among dim=0 into 2 chunks: first is completed, 2nd is cutoff by length."""
        chunks = 2
        if self.batch is not None:

            # 假设 key_name 是你要用来划分的 key
            key_name = "reason"

            mask = (self.batch[key_name] == 1).squeeze(-1)  # 变成 [32]

            # 直接使用视图，不进行复制
            batch_true = self.batch[mask]
            batch_false = self.batch[~mask]
            batchsize = batch_false.batch_size[0]

            # 计算需要保留的样本数量
            trim_size = (batchsize // world_size) * world_size

            batch_false = batch_false[:trim_size]

            batch_lst = [batch_true, batch_false] # length , finished
        else:
            batch_lst = [None for _ in range(chunks)]

        # 初始化两个非张量 batch
        non_tensor_batch_true = {}
        non_tensor_batch_false = {}
        non_tensor_batch_lst = [{} for _ in range(chunks)]

        # 遍历 self.non_tensor_batch 中的每个 key，按照 mask 进行分割
        for key, val in self.non_tensor_batch.items():
            assert isinstance(val, np.ndarray)
            
            # 按 mask 分割
            non_tensor_batch_true[key] = val[mask.cpu().numpy()]
            non_tensor_batch_false[key] = val[~mask.cpu().numpy()]
            batchsize = non_tensor_batch_false[key].shape[0]
            # 计算需要保留的样本数量
            trim_size = (batchsize // world_size) * world_size
            non_tensor_batch_false[key] = non_tensor_batch_false[key][:trim_size]
        non_tensor_batch_lst = [non_tensor_batch_true, non_tensor_batch_false] # length , finished

        output = []
        for i in range(chunks):
            output.append(
                DataProto(batch=batch_lst[i], non_tensor_batch=non_tensor_batch_lst[i], meta_info=self.meta_info))

        return output


    def chunk(self, chunks: int) -> List['DataProto']:
        """Split the batch among dim=0 into chunks. The meta_info is passed to each DataProto after split.

        Args:
            chunks (int): the number of chunks to split on dim=0

        Returns:
            List[DataProto]: a list of DataProto after splitting
        """
        assert len(
            self) % chunks == 0, f'only support equal chunk. Got size of DataProto {len(self)} and chunk {chunks}.'

        if self.batch is not None:
            batch_lst = self.batch.chunk(chunks=chunks, dim=0)
        else:
            batch_lst = [None for _ in range(chunks)]

        non_tensor_batch_lst = [{} for _ in range(chunks)]
        for key, val in self.non_tensor_batch.items():
            assert isinstance(val, np.ndarray)
            non_tensor_lst = np.array_split(val, chunks)
            assert len(non_tensor_lst) == chunks
            for i in range(chunks):
                non_tensor_batch_lst[i][key] = non_tensor_lst[i]

        output = []
        for i in range(chunks):
            output.append(
                DataProto(batch=batch_lst[i], non_tensor_batch=non_tensor_batch_lst[i], meta_info=self.meta_info))

        return output

    
    @staticmethod
    def truncate_and_filter_batches(batch_lst, maxlen, pad_id):
        mask_list = []
        for i, batch in enumerate(batch_lst):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            position_ids = batch['position_ids']
            reason = batch['reason']

            # 获取 batch_size 和 seq_len
            batch_size, seq_len = input_ids.shape

            # 截断到 maxlen
            if seq_len > maxlen:
                input_ids = input_ids[:, -maxlen:]
                attention_mask = attention_mask[:, -maxlen:]
                position_ids = position_ids[:, -maxlen:]

            # 获取每个样本的左侧第一个元素
            first_elements = input_ids[:, 0]

            # 筛选：如果左侧第一个元素不是 pad_id，就移除
            mask = first_elements == pad_id
            input_ids = input_ids[mask]
            attention_mask = attention_mask[mask]
            position_ids = position_ids[mask]
            reason = reason[mask]
            mask_list.append(mask)
            # 只有在 batch_size 仍然大于 0 时，才加入新的 batch
            if input_ids.shape[0] > 0:
                new_batch = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'position_ids': position_ids,
                    'reason': reason
                }
                batch_lst[i] = TensorDict(source=new_batch, batch_size=(input_ids.shape[0],))
            else:
                batch_lst[i] = None  # 如果 batch 被完全筛除，则设为 None
    
        # 移除 None 元素，保持 batch_lst 的结构
        batch_lst = [batch for batch in batch_lst if batch is not None]

        return batch_lst, mask_list

    @staticmethod
    def concat_partial(max_len, pad_token_id, data: List['DataProto'], world_size: int) -> 'DataProto':
        """Concat a list of DataProto. The batch is concatenated among dim=0.
        The meta_info is assumed to be identical and will use the first one.

        Args:
            data (List[DataProto]): list of DataProto

        Returns:
            DataProto: concatenated DataProto
        """
        batch_lst = []
        # 过滤掉没有 batch 的 DataProto, 针对self.replay_buffer和self.prompt_buffer
        data = [d for d in data if d is not None and d.batch is not None]
        for batch in data:
            batch_lst.append(batch.batch)

        batch_lst ,mask_lst = DataProto.truncate_and_filter_batches(batch_lst, max_len, pad_token_id)
        if batch_lst[0] is not None:
            new_batch = torch.cat(batch_lst, dim=0)
        else:
            new_batch = None

        # truncate_and_filter_batches里面去掉了partial rollout中prompt+response < max_len的batch
        # 所以这里的non_tensor_batch也要做相应的处理
        for idx, d in enumerate(data):
            assert mask_lst[idx].shape[0] == d.batch.batch_size[0], "mask shape not match"
            for key, val in d.non_tensor_batch.items():
                d.non_tensor_batch[key] = val[mask_lst[idx].cpu().numpy()]


        # hack, 原batch在前面
        non_tensor_batch = list_of_dict_to_dict_of_list(list_of_dict=[d.non_tensor_batch for d in data])
        for key, val in non_tensor_batch.items():
            non_tensor_batch[key] = np.concatenate(val, axis=0)

        if new_batch is not None:
            total_len = new_batch.shape[0]
            adjusted_len = (total_len // world_size) * world_size  # 向下取整，确保是 world_size 的整数倍

            if adjusted_len == total_len:
                remaining_batch = None
                remaining_non_tensor_batch = None
            else:
                remaining_batch = new_batch[:-adjusted_len]
                new_batch = new_batch[-adjusted_len:]

                remaining_non_tensor_batch = {key: val[:-adjusted_len] for key, val in non_tensor_batch.items()}
                non_tensor_batch = {key: val[-adjusted_len:] for key, val in non_tensor_batch.items()}

        else:
            remaining_batch = None
            remaining_non_tensor_batch = None

        adjusted_data = DataProto(batch=new_batch, non_tensor_batch=non_tensor_batch, meta_info=data[0].meta_info)
        remaining_data = DataProto(batch=remaining_batch, non_tensor_batch=remaining_non_tensor_batch, meta_info=data[0].meta_info) if remaining_batch is not None else None

        return adjusted_data, remaining_data


    @staticmethod
    def concat(data: List['DataProto']) -> 'DataProto':
        """Concat a list of DataProto. The batch is concatenated among dim=0.
        The meta_info is assumed to be identical and will use the first one.

        Args:
            data (List[DataProto]): list of DataProto

        Returns:
            DataProto: concatenated DataProto
        """
        batch_lst = []
        for batch in data:
            batch_lst.append(batch.batch)
        if batch_lst[0] is not None:
            new_batch = torch.cat(batch_lst, dim=0)
        else:
            new_batch = None
        non_tensor_batch = list_of_dict_to_dict_of_list(list_of_dict=[d.non_tensor_batch for d in data])
        for key, val in non_tensor_batch.items():
            non_tensor_batch[key] = np.concatenate(val, axis=0)
        return DataProto(batch=new_batch, non_tensor_batch=non_tensor_batch, meta_info=data[0].meta_info)

    def reorder(self, indices):
        """
        Note that this operation is in-place
        """
        indices_np = indices.detach().numpy()
        self.batch = self.batch[indices]
        self.non_tensor_batch = {key: val[indices_np] for key, val in self.non_tensor_batch.items()}

    def repeat(self, repeat_times=2, interleave=True, question_number=None):
        """
        Repeat the batch data a specified number of times.

        Args:
            repeat_times (int): Number of times to repeat the data.
            interleave (bool): Whether to interleave the repeated data.

        Returns:
            DataProto: A new DataProto with repeated data.
        """
        # 计算 a
        # print("question_number.shape", question_number)
        # a = torch.bincount(question_number.view(-1))
        # print("a.shape", a)

        # 计算变化的位置
        change_indices = torch.where(question_number[1:] != question_number[:-1])[0] + 1

        # 计算每个连续段的长度
        a = torch.diff(torch.cat((torch.tensor([0]), change_indices, torch.tensor([question_number.numel()]))))
        print("a.shape", a)
        if self.batch is not None:
            if a is not None and interleave:
                repeat_mask = a
                repeated_tensors = {
                    key: tensor.repeat_interleave(repeat_mask, dim=0) for key, tensor in self.batch.items()
                }
                repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(sum(repeat_mask),),
                )
            else:
                if interleave:
                    # Interleave the data
                    repeated_tensors = {
                        key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()
                    }
                else:
                    # Stack the data
                    repeated_tensors = {
                        key: tensor.unsqueeze(0).expand(repeat_times, *tensor.shape).reshape(-1, *tensor.shape[1:])
                        for key, tensor in self.batch.items()
                    }

                repeated_batch = TensorDict(
                    source=repeated_tensors,
                    batch_size=(self.batch.batch_size[0] * repeat_times,),
                )
        else:
            repeated_batch = None

        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            if a is not None and interleave:
                # 这里好像把question_number复制了n=gpus份
                assert val.shape[0] == a.shape[0], f"val.shape[0] != a.shape[0], {val.shape[0]} != {a.shape[0]}"
                repeated_non_tensor_batch[key] = np.repeat(val, a.cpu().numpy(), axis=0)
            else:
                if interleave:
                    repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)
                else:
                    repeated_non_tensor_batch[key] = np.tile(val, (repeat_times,) + (1,) * (val.ndim - 1))

        return DataProto(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )


import ray


@dataclass
class DataProtoFuture:
    """
    DataProtoFuture aims to eliminate actual data fetching on driver. By doing so, the driver doesn't have to wait
    for data so that asynchronous execution becomes possible. 
    DataProtoFuture contains a list of futures from another WorkerGroup of size world_size.
    - collect_fn is a Callable that reduces the list of futures to a DataProto
    - dispatch_fn is a Callable that partitions the DataProto into a list of DataProto of size world_size and then select

    Potential issue: we can optimize dispatch_fn(collect_fn) such that only needed data is fetched on destination
    - DataProtoFuture only supports directly passing from the output of a method to another input. You can't perform any
    operation on the DataProtoFuture in driver.
    """
    collect_fn: Callable
    futures: List[ray.ObjectRef]
    dispatch_fn: Callable = None

    @staticmethod
    def concat(data: List[ray.ObjectRef]) -> 'DataProtoFuture':
        output = DataProtoFuture(collect_fn=DataProto.concat, futures=data)
        return output

    def chunk(self, chunks: int) -> List['DataProtoFuture']:
        from functools import partial

        arg_future_lst = []
        for i in range(chunks):
            # note that we can't directly pass i and chunks
            def dispatch_fn(x, i, chunks):
                return x.chunk(chunks=chunks)[i]

            arg_future = DataProtoFuture(collect_fn=self.collect_fn,
                                         dispatch_fn=partial(dispatch_fn, i=i, chunks=chunks),
                                         futures=self.futures)
            arg_future_lst.append(arg_future)
        return arg_future_lst

    def get(self):
        output = ray.get(self.futures)  # dp_size.
        for o in output:
            assert isinstance(o, DataProto)
        output = self.collect_fn(output)  # select dp, concat
        if self.dispatch_fn is not None:
            output = self.dispatch_fn(output)  # split in batch dim, select using dp
        return output
