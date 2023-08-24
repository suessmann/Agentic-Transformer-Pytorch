import torch


def reduce_grouped_max(eval_returns: torch.Tensor, eval_episode_len: torch.Tensor):
    """
    A function that picks the maximum return from in-context episodes.
    This is how it is most likely done in the paper "Emergent Agentic Transformer from Chain of Hindsight Experience"
    """
    episode_return, idx_max = torch.max(eval_returns, dim=1)
    episode_len = eval_episode_len[:, idx_max]

    return episode_return.item(), episode_len.item(), idx_max


def reduce_grouped_last(eval_returns: torch.Tensor, eval_episode_len: torch.Tensor):
    """
    A function that picks the last return from in-context episodes.
    An alternative way to evaluate returns from a model.
    """
    episode_return = eval_returns[:, -1]
    episode_len = eval_episode_len[:, -1]
    idx_max = torch.tensor(3.0, device=eval_returns.device)

    return episode_return.item(), episode_len.item(), idx_max
