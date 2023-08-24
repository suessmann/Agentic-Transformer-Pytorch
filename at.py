# inspiration:
# 1. https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py  # noqa
# 2. https://github.com/karpathy/minGPT
import os
import random
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import d4rl  # noqa
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import wandb
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm, trange  # noqa
import accelerate
from accelerate.state import AcceleratorState
from eval_protocols import reduce_grouped_last, reduce_grouped_max

import time


@dataclass
class TrainConfig:
    # wandb params
    project: str = ""
    group: str = ""
    name: str = ""
    # model params
    embedding_dim: int = 128
    num_layers: int = 3
    num_heads: int = 1
    episode_len: int = 300
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    max_action: float = 1.0
    # training params
    env_name: str = "walker2d-medium-replay-v2"
    n_traj_max: int = 4
    learning_rate: float = 0.0001
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0001
    clip_grad: Optional[float] = 0.25
    batch_size: int = 2
    update_steps: int = 100_000
    warmup_steps: int = 10_000
    reward_scale: float = 0.001
    num_workers: int = 1  # TODO: DO NOT FORGET !!!!
    # evaluation params
    target_returns: Tuple[float, ...] = (3600.0, 1800.0)
    eval_protocol: str = "last"  # max or last
    eval_episodes: int = 26
    eval_every: int = 10_000
    # general params
    use_scheduler: bool = True
    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False
    train_seed: int = 10
    eval_seed: int = 42
    # distributed
    gradient_accumulation_steps: int = 1
    mixed_precision: str = None
    dynamo_backend: str = None

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        # adjust LR for multi_gpu setup
        self.learning_rate *= (
            torch.cuda.device_count() if torch.cuda.device_count() > 1 else 1
        )
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

        self.seq_len = self.episode_len * self.n_traj_max


# general utils
def set_seed(
    seed: int,
    env: Optional[gym.Env] = None,
    deterministic_torch: bool = False,
    accelerator=None,
):
    seed += AcceleratorState().process_index
    print(f"\nPID: {os.getpid()}, seed: {seed}\n")
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict, accelerator) -> None:
    init_kwargs = {
        "group": config["group"],
        "name": config["name"],
        "id": str(uuid.uuid4()),
        "save_code": True,
    }

    accelerator.init_trackers(
        project_name=config["project"],
        config=config,
        init_kwargs={"wandb": init_kwargs},
    )


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


# some utils functionalities specific for Decision Transformer
def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def calculate_new_returns(R_0, rewards):
    returns = np.zeros(len(rewards) + 1)
    returns[0] = R_0
    returns[1:] = rewards
    returns[1:] *= -1

    returns = np.cumsum(returns)
    return returns[:-1]


def load_d4rl_trajectories(
    env_name: str,
    episode_len: int,
    gamma: float = 1.0,
) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
    dataset = gym.make(env_name).get_dataset()
    traj, traj_len = [], []

    data_ = defaultdict(list)
    for i in trange(dataset["rewards"].shape[0], desc="Processing trajectories"):
        data_["observations"].append(dataset["observations"][i])
        data_["actions"].append(dataset["actions"][i])
        data_["rewards"].append(dataset["rewards"][i])

        if (
            dataset["terminals"][i]
            or dataset["timeouts"][i]
            or len(data_["rewards"]) >= episode_len
        ):
            episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
            # return-to-go if gamma=1.0, just discounted returns else
            episode_data["returns"] = discounted_cumsum(
                episode_data["rewards"], gamma=gamma
            )
            traj.append(episode_data)
            traj_len.append(episode_data["actions"].shape[0])
            # reset trajectory buffer
            data_ = defaultdict(list)

    # needed for normalization, weighted sampling, other stats can be added also
    info = {
        "obs_mean": dataset["observations"].mean(0, keepdims=True),
        "obs_std": dataset["observations"].std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_len),
    }
    return traj, info


class SequenceDataset(IterableDataset):
    def __init__(
        self,
        env_name: str,
        n_traj_max: int,
        episode_len: int,
        seq_len: int = 10,
        reward_scale: float = 1.0,
    ):
        self.dataset, info = load_d4rl_trajectories(
            env_name, episode_len=episode_len, gamma=1.0
        )
        self.reward_scale = reward_scale
        self.seq_len = seq_len
        self.n_traj_max = n_traj_max
        self.episode_len = episode_len

        self.state_mean = info["obs_mean"]
        self.state_std = info["obs_std"]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L116 # noqa
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()

    def __prepare_sample(self, traj_idx, n):
        states, actions, returns, rewards, dataset_len = [], [], [], [], []

        # extract n trajectories
        for i in traj_idx:
            traj = self.dataset[i]
            length = len(traj["rewards"])
            states.append(pad_along_axis(traj["observations"], pad_to=self.episode_len))
            actions.append(pad_along_axis(traj["actions"], pad_to=self.episode_len))

            returns.append(traj["returns"][0])
            rewards.append(traj["rewards"])
            dataset_len.append(length)

        d = np.zeros((n, self.episode_len, 1), dtype=np.float32)
        idx_sorted = np.argsort(returns)
        R_0 = returns[idx_sorted[-1]]

        ret = []
        rew = []
        mask = []

        # recalculate new returns based on R_0 and update d
        for j in idx_sorted:
            ret.append(
                pad_along_axis(
                    calculate_new_returns(R_0, rewards[j]), pad_to=self.episode_len
                )
            )
            rew.append(pad_along_axis(rewards[j], pad_to=self.episode_len))
            d[j, dataset_len[j] - 1] = 1 if returns[j] >= R_0 else 0

            mask.append(
                np.hstack(
                    [
                        np.zeros(dataset_len[j], dtype=np.float32),
                        np.full(
                            self.episode_len - dataset_len[j],
                            fill_value=float("-inf"),
                            dtype=np.float32,
                        ),
                    ]
                )
            )

        states = np.asarray(states, dtype=np.float32)[idx_sorted].reshape(
            n * self.episode_len, -1
        )
        actions = np.asarray(actions, dtype=np.float32)[idx_sorted].reshape(
            n * self.episode_len, -1
        )
        rewards = np.asarray(rew, dtype=np.float32).reshape(
            n * self.episode_len, -1
        )  # already in the right order
        returns = np.asarray(ret, dtype=np.float32).reshape(
            n * self.episode_len, -1
        )  # already in the right order
        mask = np.asarray(mask).reshape(n * self.episode_len)
        d = d.reshape(n * self.episode_len, -1)

        time_steps = np.arange(self.episode_len)
        time_steps = np.tile(time_steps, self.n_traj_max)

        states = (states - self.state_mean) / self.state_std
        returns = returns * self.reward_scale
        rewards = rewards * self.reward_scale

        # pad to the full sequence length (max(n) * episode_len)
        states = pad_along_axis(states, pad_to=(self.n_traj_max * self.episode_len))
        actions = pad_along_axis(actions, pad_to=(self.n_traj_max * self.episode_len))
        rewards = pad_along_axis(rewards, pad_to=(self.n_traj_max * self.episode_len))
        returns = pad_along_axis(returns, pad_to=(self.n_traj_max * self.episode_len))
        d = pad_along_axis(d, pad_to=(self.n_traj_max * self.episode_len))
        # padding mask with -inf since we have a b
        mask = pad_along_axis(
            mask, pad_to=(self.n_traj_max * self.episode_len), fill_value=float("-inf")
        )

        # mask to select only the last episode
        best_traj_start = (n - 1) * self.episode_len
        best_traj_finish = best_traj_start + self.episode_len
        best_traj_range = np.arange(best_traj_start, best_traj_finish, dtype=np.int32)
        return (
            states,
            actions,
            returns,
            rewards,
            d,
            time_steps,
            mask,
        ), best_traj_range

    def __iter__(self):
        while True:
            n = np.random.randint(1, self.n_traj_max)
            traj_idx = np.random.choice(len(self.dataset), size=n, p=self.sample_prob)
            # start_idx = random.randint(
            #     0, self.dataset[traj_idx]["rewards"].shape[0] - 1
            # )
            yield self.__prepare_sample(traj_idx, n)


# Transformer implementation
class TransformerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.full(size=(seq_len + 1, seq_len), fill_value=float("-inf"))
            )[1:].to(torch.float32),
        )
        #
        self.seq_len = seq_len

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]

        norm_x = self.norm1(x)
        attention_out = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        # by default pytorch attention does not use dropout
        # after final attention weights projection, while minGPT does:
        # https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py#L70 # noqa
        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        return x


class AgenticTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_traj_max: int,
        seq_len: int = 10,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        max_action: float = 1.0,
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        # self.reward_emb = nn.Linear(1, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)
        self.d_emb = nn.Linear(1, embedding_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=4 * seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.action_head = nn.Sequential(
            nn.Linear(embedding_dim, action_dim), nn.Tanh()
        )
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action
        self.n_traj_max = n_traj_max

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        states: torch.Tensor,  # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
        rewards: torch.Tensor,
        d: torch.Tensor,
        returns_to_go: torch.Tensor,  # [batch_size, seq_len]
        time_steps: torch.Tensor,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states) + time_emb
        act_emb = self.action_emb(actions) + time_emb
        # reward_emb = self.reward_emb(rewards)
        d_emb = self.d_emb(d)
        returns_emb = self.return_emb(returns_to_go) + time_emb

        # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        sequence = (
            torch.stack([returns_emb, state_emb, act_emb, d_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 4 * seq_len, self.embedding_dim)
        )
        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask] * 4, dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 4 * seq_len)
            )
        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out)
        # [batch_size, seq_len, action_dim]
        # predict actions only from state embeddings
        out = self.action_head(out[:, 1::4]) * self.max_action
        return out


# Training and evaluation logic
@torch.no_grad()
def eval_rollout(
    distributed_model: AgenticTransformer,
    env: gym.Env,
    target_return: float,
    reward_scale: float,
    accelerator,
    eps: float = 1e-4,
):
    eps *= reward_scale
    device = accelerator.device
    model = accelerator.unwrap_model(distributed_model)
    states = torch.zeros(
        1,
        model.episode_len * model.n_traj_max,
        model.state_dim,
        dtype=torch.float,
        device=device,
    )
    actions = torch.zeros(
        1,
        model.episode_len * model.n_traj_max,
        model.action_dim,
        dtype=torch.float,
        device=device,
    )
    returns = torch.zeros(
        1, model.episode_len * model.n_traj_max, 1, dtype=torch.float, device=device
    )
    rewards = torch.zeros(
        1, model.episode_len * model.n_traj_max, 1, dtype=torch.float, device=device
    )
    d = torch.zeros(
        1, model.episode_len * model.n_traj_max, 1, dtype=torch.float, device=device
    )

    time_steps = torch.zeros(
        1, model.episode_len * model.n_traj_max, dtype=torch.long, device=device
    )

    episodes_in_context = 0
    context_step = 0

    eval_returns = torch.zeros(1, model.n_traj_max, device=device)
    eval_episode_len = torch.zeros(1, model.n_traj_max, device=device)

    while episodes_in_context != model.n_traj_max:
        # reset returns after episode ends
        episode_return, episode_len = 0.0, 0.0
        states[:, context_step] = torch.as_tensor(env.reset())
        returns[:, context_step] = torch.as_tensor(target_return)

        for episode_step in range(model.episode_len):
            step = episode_step + context_step
            # first select history up to step, then select last seq_len states,
            # step + 1 as : operator is not inclusive, last action is dummy with zeros
            # (as model will predict last, actual last values are not important)
            predicted_actions = distributed_model(  # fix this noqa!!!
                states=states[:, : step + 1][:, -model.seq_len :],
                actions=actions[:, : step + 1][:, -model.seq_len :],
                rewards=rewards[:, : step + 1][:, -model.seq_len :],
                d=d[:, : step + 1][:, -model.seq_len :],
                returns_to_go=returns[:, : step + 1][:, -model.seq_len :],
                time_steps=time_steps[:, : step + 1][:, -model.seq_len :],
            )

            predicted_action = predicted_actions[0, -1].cpu().numpy()
            next_state, reward, done, info = env.step(predicted_action)
            # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
            actions[:, step] = torch.as_tensor(predicted_action)
            rewards[:, step] = torch.as_tensor(reward)
            d[:, step] = 1 if returns[:, step] - reward <= eps else 0

            episode_return += reward
            episode_len += 1

            if done or episode_step == model.episode_len - 1:
                break

            states[:, step + 1] = torch.as_tensor(next_state)
            returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)
            time_steps[:, step + 1] = episode_step + 1

        eval_returns[:, episodes_in_context] = episode_return
        eval_episode_len[:, episodes_in_context] = episode_len

        episodes_in_context += 1
        context_step += int(episode_len)

    return eval_returns, eval_episode_len


@pyrallis.wrap()
def train(config: TrainConfig):
    accelerator = accelerate.Accelerator(
        log_with="wandb",
        split_batches=True,
        device_placement=True,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        dynamo_backend=config.dynamo_backend,
        step_scheduler_with_optimizer=True,
    )

    set_seed(
        config.train_seed,
        deterministic_torch=config.deterministic_torch,
        accelerator=accelerator,
    )
    device = accelerator.device

    # data & dataloader setup
    with accelerator.main_process_first():
        dataset = SequenceDataset(
            config.env_name,
            seq_len=config.seq_len,
            reward_scale=config.reward_scale,
            n_traj_max=config.n_traj_max,
            episode_len=config.episode_len,
        )

    trainloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=config.num_workers,
    )

    # evaluation environment with state & reward preprocessing (as in dataset above)
    eval_env = wrap_env(
        env=gym.make(config.env_name),
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=config.reward_scale,
    )
    # model & optimizer & scheduler setup
    config.state_dim = eval_env.observation_space.shape[0]
    config.action_dim = eval_env.action_space.shape[0]
    model = AgenticTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        n_traj_max=config.n_traj_max,
        embedding_dim=config.embedding_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        max_action=config.max_action,
    )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda steps: min((steps + 1) / config.warmup_steps, 1),
    )

    wandb_init(asdict(config), accelerator)

    if accelerator.is_main_process:
        wandb.watch(model)

    # eval reduction to use
    if config.eval_protocol == "max":
        reduce = reduce_grouped_max
    elif config.eval_protocol == "last":
        reduce = reduce_grouped_last
    else:
        print('Choose a protocol from "max" or "last')
        exit(1)

    # save config to the checkpoint
    if accelerator.is_main_process:
        if config.checkpoints_path is not None:
            print(f"Checkpoints path: {config.checkpoints_path}")
            os.makedirs(config.checkpoints_path, exist_ok=True)
            with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
                pyrallis.dump(config, f)

        wandb.run.summary["parameters_number"] = sum(
            p.numel() for p in model.parameters()
        )
        accelerator.print(
            f"Total parameters: {sum(p.numel() for p in model.parameters())}"
        )

    trainloader, model, optim, scheduler = accelerator.prepare(
        trainloader, model, optim, scheduler
    )
    trainloader_iter = iter(trainloader)

    # training_steps_per_process = config.update_steps // accelerator.num_processes
    eval_episodes = config.eval_episodes // accelerator.num_processes
    transitions = 0
    # with accelerator.accumulate(model=model):
    for step in trange(config.update_steps, desc="Training"):
        ts_step = time.time()
        batch = next(trainloader_iter)

        (
            states,
            actions,
            returns,
            rewards,
            d,
            time_steps,
            mask,
        ), best_traj_range = batch
        padding_mask = mask

        predicted_actions = model(
            states=states,
            actions=actions,
            rewards=rewards,
            d=d,
            returns_to_go=returns,
            time_steps=time_steps,
            padding_mask=padding_mask,
        )

        best_episode_mask = (
            torch.arange(config.batch_size // accelerator.num_processes)[:, None],
            best_traj_range,
        )
        loss = F.mse_loss(
            predicted_actions[best_episode_mask],
            actions[best_episode_mask].detach(),
            reduction="none",
        )
        # [batch_size, seq_len, action_dim] * [batch_size, seq_len, 1]
        mask = F.tanh(mask[best_episode_mask]) + 1
        loss = (loss * mask.unsqueeze(-1)).mean()

        optim.zero_grad()
        accelerator.backward(loss)
        if accelerator.sync_gradients and config.clip_grad is not None:
            accelerator.unscale_gradients(optimizer=optim)
            accelerator.clip_grad_norm_(model.parameters(), config.clip_grad)

        optim.step()
        if config.use_scheduler:
            scheduler.step()

        time_step = time.time() - ts_step
        transitions += states.shape[0] * states.shape[1] * accelerator.num_processes

        if step % 100 == 0:
            loss = accelerator.gather(loss)
            accelerator.log(
                {
                    "train_loss": loss.mean().item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "transitions": transitions,
                    "times/step_time": time_step,
                },
                step=step,
            )

        # validation in the env for the actual online performance
        if step % config.eval_every == 0 or step == config.update_steps - 1:
            ts_eval = time.time()
            model.eval()
            for target_return in config.target_returns:
                eval_env.seed(config.eval_seed + AcceleratorState().process_index)
                eval_returns = []
                idx_maxs = []

                for _ in trange(eval_episodes, desc=f"Evaluation: {target_return}"):
                    eval_return, eval_len = eval_rollout(
                        distributed_model=model,
                        env=eval_env,
                        target_return=target_return * config.reward_scale,
                        accelerator=accelerator,
                        reward_scale=config.reward_scale,
                    )
                    eval_return, eval_len, idx_max = reduce(eval_return, eval_len)
                    # unscale for logging & correct normalized score computation
                    eval_returns.append(eval_return / config.reward_scale)
                    idx_maxs.append(idx_max)

                normalized_scores = (
                    eval_env.get_normalized_score(np.array(eval_returns)) * 100
                )

                normalized_scores = torch.as_tensor(normalized_scores, device=device)
                eval_returns = torch.as_tensor(eval_returns, device=device)

                eval_returns = accelerator.gather(eval_returns)
                normalized_scores = accelerator.gather(normalized_scores)
                idx_maxs = torch.cat(accelerator.gather(idx_maxs))

                time_eval = time.time() - ts_eval
                accelerator.log(
                    {
                        f"eval/{target_return}_return_mean": torch.mean(eval_returns),
                        f"eval/{target_return}_return_std": torch.std(eval_returns),
                        f"eval/{target_return}_normalized_score_mean": torch.mean(
                            normalized_scores
                        ).item(),
                        f"eval/{target_return}_normalized_score_std": torch.std(
                            normalized_scores
                        ).item(),
                        f"eval/max_reward_in_context_id": torch.mean(idx_maxs).item(),
                        f"times/eval_time": time_eval,
                    },
                    step=step,
                )

            accelerator.wait_for_everyone()
            model.train()

    accelerator.end_training()


if __name__ == "__main__":
    train()
