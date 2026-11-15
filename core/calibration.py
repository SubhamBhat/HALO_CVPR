import torch
import numpy as np
from sklearn.metrics import f1_score

def calibrate_tau_grid_search(val_logits, val_targets, class_counts, tau_range=(0.0, 3.0), step=0.01, use_prior_weights=True):
    """
    Bounded Post-Hoc Log-Prior Calibration (Proposition 3 & Algorithm 1):
      tau* = argmax_{tau in [0, lambda]} Macro-F1_bal(D_val; tau)
      Adjusts logits using the empirical class priors: z~_c = z_c - tau * log(pi_c)
      with analytical upper bound lambda = R / log(pi_max / pi_min) <= 3.0.
      Validation samples are inverse-prior weighted w_c^val = 1 / (pi_c + eps) (Eq. 20)
      to evaluate balanced macro-F1 on the long-tailed validation split.
    """
    if torch.is_tensor(val_logits):
        val_logits = val_logits.cpu().numpy()
    if torch.is_tensor(val_targets):
        val_targets = val_targets.cpu().numpy()
        
    counts = np.array(class_counts, dtype=np.float64)
    num_classes = len(counts)
    priors = counts / counts.sum()
    log_priors = np.log(priors + 1e-12)
    sw = (1.0 / (priors[val_targets] + 1e-9)) if use_prior_weights else np.ones(len(val_targets), dtype=np.float64)
    
    best_tau = 0.0
    best_f1 = -1.0
    
    for tau in np.arange(tau_range[0], tau_range[1] + step * 0.5, step):
        # Apply shift: z~_c = z_c - tau * log(pi_c)
        shifted_logits = val_logits - tau * log_priors
        preds = np.argmax(shifted_logits, axis=1)
        
        cm = np.zeros((num_classes, num_classes), dtype=np.float64)
        np.add.at(cm, (val_targets, preds), sw)
        tp = np.diag(cm)
        denom = cm.sum(axis=1) + cm.sum(axis=0)
        f1 = float(np.where(denom > 0, 2.0 * tp / denom, 0.0).mean())
        
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)
            
    return best_tau, best_f1

def apply_calibration(logits, class_counts, tau_star):
    """
    Applies the optimal tau* to inference logits to get the final predictions.
    """
    counts = np.array(class_counts, dtype=np.float64)
    priors = counts / counts.sum()
    log_priors = np.log(priors + 1e-9)
    
    if torch.is_tensor(logits):
        device = logits.device
        log_priors = torch.tensor(log_priors, dtype=torch.float32, device=device)
        return logits - tau_star * log_priors
    else:
        return logits - tau_star * log_priors


def calibrate_temperature_nll(val_logits, val_targets):
    """
    Standard post-hoc Temperature Scaling (Guo et al., ICML 2017):
    Finds scalar T* > 0 minimizing negative log-likelihood on validation logits.
    Preserves top-1 predictions (Acc and Macro-F1 unchanged) while calibrating ECE.
    """
    z_t = val_logits.detach().float().cpu() if torch.is_tensor(val_logits) else torch.tensor(val_logits, dtype=torch.float32)
    y_t = val_targets.detach().long().cpu() if torch.is_tensor(val_targets) else torch.tensor(val_targets, dtype=torch.long)
    best_T, best_nll = 1.0, float("inf")
    for T in np.concatenate([np.arange(0.2, 2.0, 0.05), np.arange(2.0, 5.05, 0.25)]):
        nll = torch.nn.functional.cross_entropy(z_t / float(T), y_t).item()
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T
