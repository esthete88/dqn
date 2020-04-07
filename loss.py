import torch
import torch.nn.functional as F


def compute_td_loss(states, actions, rewards, next_states, is_done,
                    agent, target_network, device, double_dqn, gamma=0.99):
    """Compute TD loss"""
    states = torch.tensor(states, device=device, dtype=torch.float)  # shape: [batch_size, *state_shape]
    actions = torch.tensor(actions, device=device, dtype=torch.long)  # shape: [batch_size]
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)  # shape: [batch_size]

    next_states = torch.tensor(next_states, device=device, dtype=torch.float)  # shape: [batch_size, *state_shape]
    is_done = torch.tensor(
        is_done.astype('float32'),
        device=device,
        dtype=torch.float
    )  # shape: [batch_size]
    is_not_done = 1 - is_done

    predicted_qvalues = agent(states)
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions]

    if double_dqn:
        predicted_next_qvalues = agent(next_states)
        next_actions = torch.argmax(predicted_next_qvalues, dim=1)
        target_qvalues = target_network(next_states)
        target_qvalues_for_actions = target_qvalues[range(len(next_actions)), next_actions]
        reference_qvalues_for_actions = rewards + gamma * target_qvalues_for_actions * is_not_done
    else:
        predicted_next_qvalues = target_network(next_states)
        next_state_values = torch.max(predicted_next_qvalues, dim=1).values
        reference_qvalues_for_actions = rewards + gamma * next_state_values * is_not_done

    loss = F.smooth_l1_loss(predicted_qvalues_for_actions, reference_qvalues_for_actions.detach())

    return loss
