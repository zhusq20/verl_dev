# Copyright 2025 Ziyi Qiu
# Copyright 2023 The vLLM team.
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py
import enum
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import (Callable, Deque, Dict, Iterable, List, Optional, Set,
                    Tuple, Union)

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import (Sequence, SequenceData, SequenceGroupMetadata,
                           SequenceGroup, SequenceGroupMetadataDelta, SequenceStatus)
from vllm.utils import Device, PyObjectCache
from vllm.core.scheduler import (ENABLE_ARTIFICIAL_PREEMPT, ARTIFICIAL_PREEMPTION_MAX_CNT,
                                 seq_group_metadata_builder, scheduler_running_outputs_builder,
                                 scheduled_seq_group_builder, ScheduledSequenceGroup, 
                                 Scheduler,SchedulerOutputs, SchedulerPrefillOutputs,
                                 SchedulerRunningOutputs, SchedulerSwappedInOutputs,
                                 SchedulingBudget)

class Scheduler(Scheduler):
    
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
        output_proc_callback: Optional[Callable] = None,
        partial_rollout_save_steps: Optional[int] = None,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        # Note for LoRA scheduling: the current policy is extremely
        # simple and NOT fair. It can lead to starvation of some
        # LoRAs. This should be improved in the future.
        self.lora_config = lora_config

        version = "v1"
        if self.scheduler_config.use_v2_block_manager:
            version = "v2"
        if self.scheduler_config.embedding_mode:
            version = "embedding"

        BlockSpaceManagerImpl = BlockSpaceManager.get_block_space_manager_class(
            version)

        num_gpu_blocks = cache_config.num_gpu_blocks
        if num_gpu_blocks:
            num_gpu_blocks //= pipeline_parallel_size

        num_cpu_blocks = cache_config.num_cpu_blocks
        if num_cpu_blocks:
            num_cpu_blocks //= pipeline_parallel_size

        # Create the block space manager.
        self.block_manager = BlockSpaceManagerImpl(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching)

        # Sequence groups in the WAITING state.
        # Contain new prefill or preempted requests.
        self.waiting: Deque[SequenceGroup] = deque()
        # Sequence groups in the RUNNING state.
        # Contain decode requests.
        self.running: Deque[SequenceGroup] = deque()
        # Sequence groups in the SWAPPED state.
        # Contain decode requests that are swapped out.
        self.swapped: Deque[SequenceGroup] = deque()
        # Sequence groups finished requests ids since last step iteration.
        # It lets the model know that any state associated with these requests
        # can and must be released after the current step.
        # This is used to evict the finished requests from the Mamba cache.
        self._finished_requests_ids: List[str] = list()
        # Time at previous scheduling step
        self.prev_time = 0.0
        # Did we schedule a prompt at previous step?
        self.prev_prompt = False
        # Latency of the last prompt step
        self.last_prompt_latency = 0.0
        # preemption mode, RECOMPUTE or SWAP
        self.user_specified_preemption_mode = scheduler_config.preemption_mode

        # The following field is test-only. It is used to inject artificial
        # preemption.
        self.enable_artificial_preemption = ENABLE_ARTIFICIAL_PREEMPT
        self.artificial_preempt_cnt = (ARTIFICIAL_PREEMPTION_MAX_CNT
                                       if self.enable_artificial_preemption
                                       else 0)
        self.num_cumulative_preemption: int = 0

        # Used to cache python objects
        self._seq_group_metadata_cache: List[PyObjectCache] = []
        self._scheduler_running_outputs_cache: List[PyObjectCache] = []
        self._scheduled_seq_group_cache: List[PyObjectCache] = []

        # For async output processing, we need to swap cache buffers between
        # iterations. I.e. since the output processing is lagged one step,
        # we cannot reuse the cached objects immediately when the schedule()
        # is called again, but only when schedule() is called the second time.
        self.output_proc_callback = output_proc_callback
        self.use_async_output_proc = self.output_proc_callback is not None
        self.num_cache_iters = 2 if self.use_async_output_proc else 1

        self.cache_id = 0
        for i in range(self.num_cache_iters):
            self._seq_group_metadata_cache.append(
                PyObjectCache(seq_group_metadata_builder))
            self._scheduler_running_outputs_cache.append(
                PyObjectCache(scheduler_running_outputs_builder))
            self._scheduled_seq_group_cache.append(
                PyObjectCache(scheduled_seq_group_builder))

        # For async postprocessor, the extra decode run cannot be done
        # when the request reaches max_model_len. In this case, the request
        # will be stopped during schedule() call and added to this stop list
        # for processing and deallocation by the free_finished_seq_groups()
        self._async_stopped: List[SequenceGroup] = []
        
        # For partial rollout scheduler
        self.partial_rollout_save_steps = partial_rollout_save_steps
        self.partial_rollout_mode = None
        self.sequence_group_rollout_steps: Dict = {}
        self.partial_rollout_enable = False
        self.partial: Deque[SequenceGroup] = deque()
        self.partial_rollout_blocks_to_swap_out: List[Tuple[int, int]] = []
        # For overlapping rollout and inference
        self.fused_request_ids = []
        
    def clear_rollout_steps(self) -> None:
        self.sequence_group_rollout_steps = {}
        
    def set_partial_rollout_enable(self, partial_rollout_enable) -> None:
        self.partial_rollout_enable = partial_rollout_enable

    def add_rollout_steps(self, request_id: Union[str, Iterable[str]]) -> None:
        if request_id in self.sequence_group_rollout_steps:
            self.sequence_group_rollout_steps[request_id] += 1
        else:
            self.sequence_group_rollout_steps[request_id] = 1

    def get_rollout_steps(self, request_id: Union[str, Iterable[str]]) -> None:
        if not request_id in self.sequence_group_rollout_steps:
            return 0
        return self.sequence_group_rollout_steps[request_id]
    
    def transfer_partial_rollout_requests(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        waiting_groups: List[SequenceGroup] = []
        for state_queue in [self.waiting, self.running, self.swapped]:
            temp_groups: List[SequenceGroup] = []
            for seq_group in state_queue:
                if not request_ids:
                    # Using 'break' here may add two extra iterations,
                    # but is acceptable to reduce complexity.
                    break
                if seq_group.request_id in request_ids:
                    if not seq_group.is_finished():
                        waiting_groups.append(seq_group)
                        temp_groups.append(seq_group)
                    request_ids.remove(seq_group.request_id)
            for temp_group in temp_groups:
                # Remove the sequence group from the state queue.
                state_queue.remove(temp_group)
        for i in range(len(waiting_groups)-1, -1, -1):
            waiting_group = waiting_groups[i]
            if waiting_group.is_finished():
                del waiting_groups[i]
                continue
            if self.partial_rollout_mode == "recompute":
                self._free_seq_group_cross_attn_blocks(waiting_group)
            else: # swap out
                self._swap_out(waiting_group, self.partial_rollout_blocks_to_swap_out)
            # Remove the request.
            for seq in waiting_group.get_seqs():
                if seq.is_finished():
                    continue
                if self.partial_rollout_mode == "recompute": # recompute
                    seq.status = SequenceStatus.WAITING
                    seq.reset_state_for_recompute()
                    self.free_seq(seq)
                else: # swap out
                    seq.status = SequenceStatus.SWAPPED
        
        self.partial.extend(waiting_groups)

    def has_unfinished_seqs(self) -> bool:
        return len(self.waiting) != 0 or len(self.running) != 0 or len(
            self.swapped) != 0

    def transfer_partial_to_waiting(self) -> None:
        self.waiting.extend(self.partial)
        self.partial.clear()
        
    def transfer_partial_to_swapped(self) -> None:
        self.swapped.extend(self.partial)
        self.partial.clear()
    
    def transfer_partial_to_running(self) -> None:
        self.running.extend(self.partial)
        self.partial.clear()

    def sorted_partial_seq_groups(self) -> List[SequenceGroup]:
        # NOTE: this function will clear self.partial
        ret = sorted(list(self.partial), key=lambda x: int(x.request_id))
        self.partial.clear()
        return ret

    def add_partial_seq_group(self, seq_group: SequenceGroup) -> None:
        self.partial.append(seq_group)
        
    def set_partial_rollout_mode(self, partial_rollout_mode: Optional[str]) -> None:
        self.partial_rollout_mode = partial_rollout_mode

    def get_and_reset_partial_rollout_blocks_to_swap_out(self) -> List[Tuple[int, int]]:
        ret = self.partial_rollout_blocks_to_swap_out
        self.partial_rollout_blocks_to_swap_out = list()
        return ret
    
    def add_fused_request_id(self, request_id: str) -> None:
        self.fused_request_ids.append(request_id)
    
    def transfer_fused_requests(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        waiting_groups: List[SequenceGroup] = []
        for state_queue in [self.waiting, self.running, self.swapped]:
            temp_groups: List[SequenceGroup] = []
            for seq_group in state_queue:
                if not request_ids:
                    # Using 'break' here may add two extra iterations,
                    # but is acceptable to reduce complexity.
                    break
                if seq_group.request_id in request_ids:
                    if not seq_group.is_finished():
                        waiting_groups.append(seq_group)
                        temp_groups.append(seq_group)
                    request_ids.remove(seq_group.request_id)
            for temp_group in temp_groups:
                # Remove the sequence group from the state queue.
                state_queue.remove(temp_group)
        for i in range(len(waiting_groups)):
            waiting_group = waiting_groups[i]
            if self.partial_rollout_mode == "recompute":
                self._free_seq_group_cross_attn_blocks(waiting_group)
            # Remove the request.
            for seq in waiting_group.get_seqs():
                if seq.is_finished():
                    continue
                self.free_seq(seq)
                
    def is_request_fused(self, request_id: str) -> bool:
        return (request_id in self.fused_request_ids)
