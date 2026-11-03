import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicWeightManager:
    """
    HALO Dynamic Weight Manager (Algorithm 1):
      - W_rep^(c) = N_max / N_c  (Linear inverse-frequency, beta=1.0)
      - W_pen^(c) = sqrt(N_max / N_c)  (Square-root dampened penalty, beta=0.5)
      - F1_tr initialized to 1.0 for all classes
      - Simple EMA: F1_smooth = 0.9 * F1_smooth + 0.1 * F1_tr
      - M_c = (max_j F1_smooth^(j) + eps) / (F1_smooth^(c) + eps)
      - W_dyn = W_pen * (1 + log(max(M_c, 1.0)))
      - W_act = (1 - gamma) * W_rep + gamma * W_dyn
      - L1 trace normalization: w = K * w / sum(w)
    """
    def __init__(self, counts=None, use_ema=True):
        if counts is None:
            try:
                import halo_unified_training as H
                counts = H.cfg.CLASS_COUNTS
            except Exception:
                counts = [5000, 5000, 5000, 5000, 5000, 500, 500, 500, 100, 100]
        self.c = len(counts)
        self.counts = np.array(counts, dtype=np.float64)
        self.use_ema = use_ema

        # Linear inverse frequency for early representation learning (beta = 1.0)
        self.w_rep = self.counts.max() / np.maximum(self.counts, 1.0)

        # Square-root dampened base penalty (beta = 0.5)
        self.w_pen = np.sqrt(self.counts.max() / np.maximum(self.counts, 1.0))
        self.w_pen_base = self.w_pen  # alias for backward compatibility
        self.w_dyn = self.w_pen.copy()
        self.w_dynamic = self.w_dyn   # alias for backward compatibility

        # Initialized to 1.0 for all classes per Algorithm 1
        self.f1_smooth = np.ones(self.c, dtype=np.float64)
        self.f_smooth = self.f1_smooth  # alias for backward compatibility
        self.initialized = False
        self.wd_history = []

    def _norm(self, w):
        return self.c * w / (w.sum() + 1e-9)

    def update(self, train_f1, train_ece=None):
        """Update dynamic weights using training set per-class F1."""
        self.initialized = True
        train_f1 = np.array(train_f1, dtype=np.float64)

        if self.use_ema:
            self.f1_smooth = 0.9 * self.f1_smooth + 0.1 * train_f1
        else:
            self.f1_smooth = train_f1.copy()
        self.f_smooth = self.f1_smooth

        eps = 1e-9
        f1_max = self.f1_smooth.max()
        M_c = (f1_max + eps) / (self.f1_smooth + eps)
        dynamic_multiplier = 1.0 + np.log(np.maximum(M_c, 1.0))

        self.w_dyn = self.w_pen * dynamic_multiplier
        self.w_dynamic = self.w_dyn

    def get_active_weights(self, gamma=0.0):
        if not self.initialized:
            normed_w = self._norm(self.w_rep)
        else:
            w_active = (1.0 - gamma) * self.w_rep + gamma * self.w_dyn
            normed_w = self._norm(w_active)
        self.wd_history.append(normed_w.copy())
        return normed_w

    def get(self, gamma=0.0):
        return self.get_active_weights(gamma)


class RLPenaltyScheduler:
    """
    Contextual Bandit for calibration penalty scheduling (Ablation baseline).
    State:  [F1_1..F1_K, ECE_1..ECE_K, epoch/max_epoch]
    Action: lambda in {0, 0.01, 0.03, 0.05, 0.1}
    Reward: Delta F1 (change in validation Macro-F1)
    """
    def __init__(self, actions=None, num_classes=10, epochs=100, lr=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions if actions is not None else [0.0, 0.01, 0.03, 0.05, 0.1]
        self.n_actions = len(self.actions)
        self.state_dim = 2 * num_classes + 1
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.W = np.zeros((self.n_actions, self.state_dim))
        self.prev_state = None
        self.prev_action_idx = None
        self.prev_f1 = 0.0
        self.lambda_history = []

    def get_state(self, f1s, eces, epoch):
        return np.concatenate([f1s, eces, [epoch / self.epochs]])

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            idx = np.random.randint(self.n_actions)
        else:
            q_values = self.W @ state
            idx = np.argmax(q_values)
        self.lambda_history.append(self.actions[idx])
        return idx, self.actions[idx]

    def update(self, state, f1):
        reward = f1 - self.prev_f1
        if self.prev_state is not None:
            q_next = np.max(self.W @ state)
            target = reward + self.gamma * q_next
            q_prev = self.W[self.prev_action_idx] @ self.prev_state
            self.W[self.prev_action_idx] += self.lr * (target - q_prev) * self.prev_state
        self.prev_f1 = f1


class HALOLoss(nn.Module):
    """
    HALO Loss (Algorithm 1):
      Weighted cross-entropy loss without heuristic label smoothing:
        ce = CrossEntropy(logits, targets, weight=w_active)
    """
    def __init__(self):
        super(HALOLoss, self).__init__()

    def forward(self, logits, targets, w_active=None, gamma=0.0):
        if w_active is not None:
            ce = F.cross_entropy(logits, targets, reduction='none')
            w = w_active[targets]
            return (ce * w).mean()
        else:
            return F.cross_entropy(logits, targets)


# Backward-compatibility alias
DCALoss = HALOLoss
