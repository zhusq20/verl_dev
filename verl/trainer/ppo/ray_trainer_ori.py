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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""
import copy
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy

import ray
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tensordict import TensorDict
import torch
WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REMAX = 'remax'
    RLOO = 'rloo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get('GPU', 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster"
                )


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]
    print(f'response_mask.shape: {attention_mask.shape}, {response_mask.shape}')

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        print(f'old_log_probs: {data.batch["old_log_probs"].shape}, ref_log_prob: {data.batch["ref_log_prob"].shape}')
        print(f'kld: {kld.shape}', 'response_mask:', response_mask.shape)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, \
                "validation gen temperature should be greater than 0 when enabling do_sample"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         processor=self.processor,
                                         prompt_key=self.config.data.prompt_key,
                                         image_key=self.config.data.get('image_key', 'images'),
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         max_response_length=self.config.data.max_response_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation=self.config.data.get('truncation', 'error'),
                                         filter_overlong_prompts=self.config.data.filter_overlong_prompts)
        assert self.train_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.train_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.train_batch_size,
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       processor=self.processor,
                                       prompt_key=self.config.data.prompt_key,
                                       image_key=self.config.data.get('image_key', 'images'),
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       max_response_length=self.config.data.max_response_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation=self.config.data.get('truncation', 'error'),
                                       filter_overlong_prompts=self.config.data.filter_overlong_prompts)
        assert self.val_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.val_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }
            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        # num_gpus = self.config.num_gpus

        tensors = {}
        non_tensors = {}
        meta_info = active_batch.meta_info

        print("line774", active_batch.batch.keys(), active_batch.non_tensor_batch.keys())
        num_gpus = self.actor_rollout_wg.world_size
        if num_gpus <= 1:
            print("line777 _generate_with_gpu_padding", active_batch.batch.keys(), active_batch.non_tensor_batch.keys())
            return self.actor_rollout_wg.generate_sequences_partial(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            print("line784 _generate_with_gpu_padding", active_batch.batch.keys(), active_batch.non_tensor_batch.keys())
            return self.actor_rollout_wg.generate_sequences_partial(active_batch)
            
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            tensors[k] = torch.cat([v, pad_sequence], dim=0)
        
        for k, v in active_batch.non_tensor_batch.items():
            if isinstance(v, np.ndarray):
                pad_sequence = np.repeat(v[0:1], padding_size, axis=0)
                non_tensors[k] = np.concatenate([v, pad_sequence], axis=0)
            else:
                non_tensors[k] = v
 
        # for key, val in data.items():
        #     if isinstance(val, torch.Tensor):
        #         tensors[key] = val
        #     elif isinstance(val, np.ndarray):
        #         non_tensors[key] = val
        #     else:
        #         raise ValueError(f'Unsupported type in data {type(val)}')

        padded_active_batch = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)


        # padded_active_batch = DataProto.from_dict(padded_batch)
        
        # Generate with padded batch
        print("line808 _generate_with_gpu_padding", padded_active_batch.batch.keys(), padded_active_batch.non_tensor_batch.keys())
        padded_output = self.actor_rollout_wg.generate_sequences_partial(padded_active_batch)
        
        # Remove padding from output
        print("padded_output.batch", padded_output.batch)
        fields = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        bsz = fields['input_ids'].shape[0]
        # 构建 TensorDict
        
        trimmed_batch = TensorDict(fields, batch_size=[bsz], device=torch.device("cpu"))
        # trimmed_batch = padded_output.batch.apply(lambda x: x[:-padding_size])
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                elif isinstance(v, np.ndarray):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        if hasattr(padded_output, 'non_tensor_batch') and padded_output.non_tensor_batch:
            # non_tensor_batch_trimmed = {}
            for k, v in padded_output.non_tensor_batch.items():
                if isinstance(v, np.ndarray):
                    padded_output.non_tensor_batch[k] = v[:-padding_size]
                else:
                    padded_output.non_tensor_batch[k] = v
        padded_output.batch = trimmed_batch
        return padded_output

    def _partial_rollout_add_from_dataloader_for_rollout(self, data_iter, replay_gen_batch):
        def get_batch_or_exhausted(data_iter, desired_size):
            batches = []
            accumulated_len = 0
            while accumulated_len < desired_size:
                try:
                    batch_dict = next(data_iter)
                    new_batch = DataProto.from_single_dict(batch_dict)
                    batches.append(new_batch)
                    accumulated_len += len(new_batch)
                except StopIteration:
                    break
            return DataProto.concat(batches) if len(batches) else None

        replay_partial_size = len(replay_gen_batch) if replay_gen_batch else 0
        n = self.config.actor_rollout_ref.rollout.n
        desired_num = self.config.data.train_batch_size - replay_partial_size // n
        new_batch = get_batch_or_exhausted(data_iter, desired_num)
        if new_batch is not None:
            # successfully added more data from dataloader
            # add metadata: is_partial and uid
            new_batch.batch["is_partial"] = torch.zeros((len(new_batch),), dtype=torch.bool)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))],
                dtype=object,
            )
            # merge with replay_gen_batch
            if replay_gen_batch is None:
                batch = new_batch
            else:
                # NOTE: here we put new_batch later, because if we clip the gen_batch
                # we deprioritize clipping partial rollout, so that its different tokens are
                # generated by the nearmost weights.
                for k, v in replay_gen_batch.batch.items():
                    if k != "is_partial":
                        if len(replay_gen_batch.batch[k].shape) > 1:
                            if replay_gen_batch.batch[k].shape[1] > self.config.data.max_prompt_length:
                                replay_gen_batch.batch[k] = v[:, -self.config.data.max_prompt_length:]
                batch = DataProto.concat([replay_gen_batch, new_batch])
        elif replay_partial_size > 0:
            # dataloader exhausted, use the remaining partial rollout for generation
            batch = replay_gen_batch
        else:
            # dataloader exhausted and no replay_gen_batch
            batch = None
        return batch, new_batch

    def _split_rollout_by_partial(self, batch: DataProto):
        tensor_batch = batch.batch
        partial_index = batch.batch["is_partial"].nonzero(as_tuple=True)[0]
        full_index = torch.logical_not(batch.batch["is_partial"]).nonzero(as_tuple=True)[0]
        partial_batch = self._data_proto_item_to_data_proto(batch[partial_index])
        full_batch = self._data_proto_item_to_data_proto(batch[full_index])
        return partial_batch, full_batch
    
    def _data_proto_item_to_data_proto(self, item):
        return DataProto(item.batch, item.non_tensor_batch, item.meta_info)

    def _move_padding_to_left(self, tensor: torch.Tensor, pad_token_id: int, prompt: torch.Tensor) -> torch.Tensor:
        # mask prompt-side padding (not to be touched)
        pad_left = (prompt == pad_token_id)
        
        # all padding
        pad_mask = (tensor == pad_token_id)
        
        # true padding: padding tokens outside prompt
        right_pad_mask = pad_mask & (~pad_left)
        
        # non-padding tokens that should remain (including prompt tokens)
        keep_mask = ~right_pad_mask  # [B, T]

        # Count how many padding tokens should move to the left
        pad_counts = right_pad_mask.sum(dim=1)  # [B]

        # Sort each row so that padding tokens (False in keep_mask) come first
        # use descending sort on keep_mask to push True (non-padding) to the right
        sorted_mask, sorted_indices = keep_mask.sort(dim=1, descending=True)  # True (keep) at right

        # Gather tokens according to sorted indices
        sorted_tensor = torch.gather(tensor, dim=1, index=sorted_indices)

        return sorted_tensor

    def _post_process_partial_rollout(self, partial_rollouts: DataProto,
                                      replay_buffer_partial: DataProto):
        """post-process partial rollouts."""
        pad_token_id: int = self.tokenizer.pad_token_id
        # 1. gen_batch_output introduced new keys, we need to remove them
        prompts_and_response = partial_rollouts.pop(["prompts", "responses"])
        prompts = prompts_and_response.batch["prompts"]
        # 2. The value is current both left and right padded. We need to make it instead
        # only right padded. This applies for "input_ids", "attention_mask", "position_ids"
        for key in ["input_ids", "attention_mask", "position_ids"]:
            # tensor = partial_rollouts.batch[key]
            tensor = partial_rollouts.batch["input_ids"]
            # locate the pad token length at the end. Note that we don't count pad token at the beginning
            pad_length_full = (tensor == pad_token_id).sum(dim=-1, keepdim=True)
            # use the first-nonzero
            pad_length_left = (prompts == pad_token_id).sum(dim=-1, keepdim=True)
            pad_length_right = pad_length_full - pad_length_left
            # update the tensor by moving all padding to the left
            batch_size, seq_len = tensor.shape
            j = torch.arange(seq_len, device=tensor.device).unsqueeze(0).expand(batch_size, seq_len)
            assert ((pad_length_left + pad_length_right) <= seq_len).all(), \
            f"Padding lengths exceed sequence length! left: {pad_length_left.max()}, right: {pad_length_right.max()}, seq_len: {seq_len}"
            indices = torch.where(
                j < pad_length_left,
                j,
                torch.where(
                    j < pad_length_left + pad_length_right,
                    j - pad_length_left + (seq_len - pad_length_right),
                    j - pad_length_right
                )
            )
            # Rearrange each row using the computed indices.
            # indices = indices.clamp(0, seq_len - 1)  # 防止越界
            tensor1 = torch.gather(partial_rollouts.batch[key], 1, indices)
            partial_rollouts.batch[key] = tensor1
            # assert (partial_rollouts.batch[key].shape[1] == 512), f"tensor shape {key=}, {partial_rollouts.batch[key].shape} does not match seq_len {512}"

        # 3. concatenate with existing replay buffer partial (Should not exist?)
        if replay_buffer_partial is None:
            replay_buffer_partial = partial_rollouts
        else:
            replay_buffer_partial = DataProto.concat([replay_buffer_partial, partial_rollouts])
        return replay_buffer_partial



    def _post_process_partial_rollout_siqi(self, partial_rollouts: DataProto,
                                      replay_buffer_partial: DataProto):
        """post-process partial rollouts."""
        # 1. concatenate with existing replay buffer partial (Should not exist?)
        if replay_buffer_partial is None:
            replay_buffer_partial = partial_rollouts
        else:
            replay_buffer_partial = DataProto.concat([replay_buffer_partial, partial_rollouts])
        return replay_buffer_partial

    def fit_yh(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):

            use_partial_rollout = getattr(self.config.actor_rollout_ref.rollout, "use_partial_rollout", False)
            if use_partial_rollout:
                assert (self.config.algorithm.adv_estimator
                        in ["gae", "reinforce_plus_plus"]), "partial rollout does not support adv_algo needs all samples"

                replay_buffer_partial: DataProto = None
                replay_buffer_full: DataProto = None
                data_iter = iter(self.train_dataloader)

                epoch_done: bool = False
                n = self.config.actor_rollout_ref.rollout.n
                while not epoch_done:
                    metrics = {}
                    timing_raw = {}
                    # start a new iteration.
                    replay_partial_size = len(replay_buffer_partial) if replay_buffer_partial else 0
                    replay_full_size = len(replay_buffer_full) if replay_buffer_full else 0

                    do_rollout_gen = False
                    new_batch = None
                    batch = None
                    # there is no enough data remain to be used for training, need to generate new rollouts
                    if replay_full_size < self.config.data.train_batch_size * n:
                        replay_gen_batch = replay_buffer_partial

                        # TODO: the train_batch_size here should be changed to another arg specialized for this.
                        if replay_partial_size < self.config.data.train_batch_size * n:
                            batch, new_batch = self._partial_rollout_add_from_dataloader_for_rollout(
                                data_iter, replay_gen_batch
                            )
                            if not batch:
                                epoch_done = True
                        else:
                            batch = replay_gen_batch
                            for k, v in batch.batch.items():
                                if k != "is_partial":
                                    if len(batch.batch[k].shape) > 1:
                                        if batch.batch[k].shape[1] > self.config.data.max_prompt_length:
                                            batch.batch[k] = v[:, -self.config.data.max_prompt_length:]

                        replay_buffer_partial = None    # replay buffer is consumed into the batch

                        if len(batch) > 0:
                            # print(f"batch.size line1023: {len(batch)}, {batch.batch['input_ids'].shape}")
                            # handle batch not divisible by world_size
                            batch, gen_batch_pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)
                            # Force it to only sample once to reduce the cost. TODO: maybe extend is_partial to int, and handle
                            # the special padding case so that it is not even computed?
                            print(f"gen_batch_pad_size: {gen_batch_pad_size}")
                            if gen_batch_pad_size > 0:
                                batch.batch["is_partial"][-gen_batch_pad_size:] = torch.ones((gen_batch_pad_size,), dtype=torch.bool)

                            gen_batch = batch.pop(
                                batch_keys=["input_ids", "attention_mask", "position_ids", "is_partial"],
                            )
                        do_rollout_gen = len(batch) > 0

                    with _timer("step", timing_raw):
                        # generate a batch
                        with _timer("gen", timing_raw):
                            if do_rollout_gen:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(
                                    gen_batch
                                )
                        if self.config.algorithm.adv_estimator == "remax":
                            with _timer("gen_max", timing_raw):
                                if do_rollout_gen:
                                    gen_baseline_batch = deepcopy(gen_batch)
                                    gen_baseline_batch.meta_info["do_sample"] = False
                                    gen_baseline_output = (
                                        self.actor_rollout_wg.generate_sequences(
                                            gen_baseline_batch
                                        )
                                    )

                                    batch = batch.union(gen_baseline_output)
                                    reward_baseline_tensor = self.reward_fn(batch)
                                    reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                                    batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                                    batch.batch["reward_baselines"] = reward_baseline_tensor

                                    del gen_baseline_batch, gen_baseline_output

                        # Now that all generation steps are done. We handle the gen_batch:
                        # If it is partial, we put it back to the replay_buffer_partial.
                        with _timer("gen_post_process", timing_raw):
                            if do_rollout_gen:
                                gen_batch_output = unpad_dataproto(gen_batch_output, pad_size=gen_batch_pad_size)

                                # handle new_batch with n > 1
                                if new_batch is not None:
                                    new_batch = new_batch.repeat(
                                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                                        interleave=True,
                                    )
                                    if replay_gen_batch is not None:
                                        batch = DataProto.concat([replay_gen_batch, new_batch])
                                    else:
                                        batch = new_batch
                                else:
                                    batch = replay_gen_batch
                                # print(f"batch.size line1083:{len(batch)}, {len(gen_batch_output)}, {gen_batch_output.batch['input_ids'].shape}")
                                batch = batch.union(gen_batch_output)
                                # print(f"batch.size line1085: {len(batch)}, {batch.batch['input_ids'].shape}")
                        # apply replay_buffer_full
                        if replay_full_size > 0:
                            batch = DataProto.concat([replay_buffer_full, batch])
                        # split batch into partial(unfinished) and full rollout
                        partial_rollouts, full_rollouts = self._split_rollout_by_partial(batch)
                        replay_buffer_partial = self._post_process_partial_rollout(
                            partial_rollouts, replay_buffer_partial,
                        )
                        batch = full_rollouts
                        # print(f"batch.size line1095: {len(batch)}, {batch.batch['input_ids'].shape}")
                        # if the batch is too large: put the rest into the replay_buffer_full
                        chunked_size = len(batch) - self.config.data.train_batch_size * n
                        if chunked_size > 0:
                            # NOTE: __getitem__ returns DataProtoItem. we need to convert it to DataProto
                            replay_buffer_full = self._data_proto_item_to_data_proto(batch[-chunked_size:])
                            batch = self._data_proto_item_to_data_proto(batch[:-chunked_size])
                        elif chunked_size < 0:
                            continue

                        # balance the number of valid tokens on each dp rank.
                        # Note that this breaks the order of data inside the batch.
                        # Please take care when you implement group based adv computation such as GRPO and rloo
                        self._balance_batch(batch, metrics=metrics)

                        # compute global_valid tokens
                        batch.meta_info["global_token_num"] = torch.sum(
                            batch.batch["attention_mask"], dim=-1
                        ).tolist()

                        # recompute old_log_probs
                        with _timer("old_log_prob", timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            batch = batch.union(old_log_prob)

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer("ref", timing_raw):
                                print(f"batch.keys: {batch.batch.keys()}")
                                # for k, v in batch.batch.items():
                                    # if k != "is_partial":
                                    #     assert v.shape[1] == 512, f"tensor shape {k=}, {v.shape} does not match seq_len {512}"
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                    batch
                                )
                                print(f"ref_log_prob: {ref_log_prob.batch['ref_log_prob'].shape}")
                                batch = batch.union(ref_log_prob)

                        # compute values
                        if self.use_critic:
                            with _timer("values", timing_raw):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)

                        with _timer("adv", timing_raw):
                            # compute scores. Support both model and function-based.
                            # We first compute the scores using reward model. Then, we call reward_fn to combine
                            # the results from reward model and rule-based results.
                            if self.use_rm:
                                # we first compute reward model score
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                            # we combine with rule-based rm
                            reward_tensor = self.reward_fn(batch)
                            batch.batch["token_level_scores"] = reward_tensor

                            # compute rewards. apply_kl_penalty if available
                            if not self.config.actor_rollout_ref.actor.get(
                                "use_kl_loss", False
                            ):
                                batch, kl_metrics = apply_kl_penalty(
                                    batch,
                                    kl_ctrl=self.kl_ctrl,
                                    kl_penalty=self.config.algorithm.kl_penalty,
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch[
                                    "token_level_scores"
                                ]

                            # compute advantages, executed on the driver process
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                            )

                        # update critic
                        if self.use_critic:
                            with _timer("update_critic", timing_raw):
                                critic_output = self.critic_wg.update_critic(batch)
                            critic_output_metrics = reduce_metrics(
                                critic_output.meta_info["metrics"]
                            )
                            metrics.update(critic_output_metrics)

                        # implement critic warmup
                        if self.config.trainer.critic_warmup <= self.global_steps:
                            # update actor
                            with _timer("update_actor", timing_raw):
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(
                                actor_output.meta_info["metrics"]
                            )
                            metrics.update(actor_output_metrics)

                        # validate
                        if (
                            self.val_reward_fn is not None
                            and self.config.trainer.test_freq > 0
                            and self.global_steps % self.config.trainer.test_freq == 0
                        ):
                            with _timer("testing", timing_raw):
                                val_metrics: dict = self._validate()
                            metrics.update(val_metrics)

                        if (
                            self.config.trainer.save_freq > 0
                            and self.global_steps % self.config.trainer.save_freq == 0
                        ):
                            with _timer("save_checkpoint", timing_raw):
                                self._save_checkpoint()

                    # collect metrics
                    metrics.update(
                        compute_data_metrics(batch=batch, use_critic=self.use_critic)
                    )
                    metrics.update(
                        compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                    )

                    # TODO: make a canonical logger that supports various backend
                    logger.log(data=metrics, step=self.global_steps)

                    self.global_steps += 1

                    if self.global_steps >= self.total_training_steps:

                        # perform validation after training
                        if self.val_reward_fn is not None:
                            val_metrics = self._validate()
                            pprint(f"Final validation metrics: {val_metrics}")
                            logger.log(data=val_metrics, step=self.global_steps)
                        if (
                            self.config.trainer.save_freq > 0
                            and (self.global_steps - 1) % self.config.trainer.save_freq != 0
                        ):
                            with _timer("save_checkpoint", timing_raw):
                                self._save_checkpoint()
                        return

                continue
            # End of partial rollout


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        replay_buffer_partial: DataProto = None # 存储partial rollout的部分
        replay_buffer_full: DataProto = None # 存储完整的rollout，但还没有用到的部分
        use_partial_rollout = getattr(self.config.actor_rollout_ref.rollout, 'use_partial_rollout', False)
        for epoch in range(self.config.trainer.total_epochs):

            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                # start a new iteration.
                replay_partial_size = len(replay_buffer_partial) if replay_buffer_partial else 0
                replay_full_size = len(replay_buffer_full) if replay_buffer_full else 0
                

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                if use_partial_rollout:
                    batch.batch['position_ids'] = torch.cat([batch.batch['position_ids'], \
                                                            torch.zeros((batch.batch['position_ids'].shape[0], \
                                                                        self.config.data.max_response_length), dtype=torch.long)], dim=-1)
                    batch.batch['attention_mask'] = torch.cat([batch.batch['attention_mask'], \
                                                            torch.zeros((batch.batch['attention_mask'].shape[0], \
                                                                        self.config.data.max_response_length), dtype=torch.long)], dim=-1)
                    batch.batch['input_ids'] = torch.cat([batch.batch['input_ids'], \
                                                        torch.zeros((batch.batch['input_ids'].shape[0], \
                                                                    self.config.data.max_response_length), dtype=torch.long)], dim=-1)
                print("replay_partial_size", replay_partial_size)
                print("replay_full_size", replay_full_size)

                if replay_partial_size > 0 and use_partial_rollout:
                    # assert 'prompts' not in replay_buffer_partial.batch.keys(), \
                    #     f"prompts should not be in replay_buffer_partial keys {replay_buffer_partial.batch.keys()}"
                    # assert batch.batch.keys() == replay_buffer_partial.batch.keys(), \
                    #     f"batch keys {batch.batch.keys()} does not match replay_buffer_partial keys {replay_buffer_partial.batch.keys()}"
                    # print(f"batch keys {batch.non_tensor_batch.keys()} does not match replay_buffer_partial keys {replay_buffer_partial.non_tensor_batch.keys()}")
                    # print("len(batch.non_tensor_batch['raw_prompt_ids'][0])", len(batch.non_tensor_batch['raw_prompt_ids'][0]), len(batch.non_tensor_batch['raw_prompt_ids'][1]))

                    # assert len(batch.non_tensor_batch['raw_prompt_ids'][0]) != len(batch.non_tensor_batch['raw_prompt_ids'][1]), \
                    #     f"raw_prompt_ids should not be the same {len(batch.non_tensor_batch['raw_prompt_ids'][0])}"
                    # for item in batch.non_tensor_batch['raw_prompt_ids']:
                        # print("batch length",len(item))
                    # for item, response in zip(replay_buffer_partial.non_tensor_batch['raw_prompt_ids'], replay_buffer_partial.non_tensor_batch['raw_response_ids']):
                        # print("replay_buffer length", len(item), len(response))
                    batch = DataProto.concat([batch, replay_buffer_partial])
                # pop those keys for generation
                if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids','raw_response_ids'],
                    )
                print("line 1350 gen_batch", gen_batch.batch.keys(), gen_batch.non_tensor_batch.keys())

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        # gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        gen_batch_output = self._generate_with_gpu_padding(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # apply replay_buffer_full
                    if use_partial_rollout:
                        if replay_full_size > 0:
                            batch = DataProto.concat([replay_buffer_full, batch])
                        # split batch into partial(unfinished) and full rollout
                        partial_rollouts, full_rollouts = self._split_rollout_by_partial(batch)
                        partial_rollouts.pop(batch_keys=['prompts', 'responses', 'is_partial'], non_tensor_batch_keys=['uid'])
                        replay_buffer_partial = self._post_process_partial_rollout_siqi(
                            partial_rollouts, replay_buffer_partial,
                        )
                        print("line1390 replay_buffer_partial.keys()", replay_buffer_partial.batch.keys(), replay_buffer_partial.non_tensor_batch.keys())
                        batch = full_rollouts
                    
                        # if the batch is too large: put the rest into the replay_buffer_full
                        chunked_size = len(batch) - self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                        if chunked_size > 0:
                            # NOTE: __getitem__ returns DataProtoItem. we need to convert it to DataProto
                            replay_buffer_full = self._data_proto_item_to_data_proto(batch[-chunked_size:])
                            batch = self._data_proto_item_to_data_proto(batch[:-chunked_size])
                        elif chunked_size < 0:
                            print("FBI WARNING !!! chunked_size < 0 !!!")
                            if replay_buffer_full is None:
                                replay_buffer_full = batch
                            else:
                                replay_buffer_full = DataProto.concat([replay_buffer_full, batch])
                            continue
                    print("finished gen")
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and ( is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    return

                self.global_steps += 1
