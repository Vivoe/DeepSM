import numpy as np
import pandas as pd
from scipy import optimize, special
from sklearn import metrics

def optimize_threshold_f1(outputs, labels):
    std = np.std(outputs)
    bounds = np.array([np.min(outputs), np.max(outputs)]) / std
    def fn(thresh):
        return -metrics.f1_score(labels, outputs >= std * thresh)
    res = optimize.minimize_scalar(fn, bounds=bounds)
    print("F1 score:", -res.fun, "Threshold:", res.x)
    return res.x

def optimize_threshold_count(outputs, labels, target):
    std = np.std(outputs)
    bounds = np.array([np.min(outputs), np.max(outputs)]) / std

    song_len = len(outputs) * 512 / 44100 / 60
    def fn(thresh):
        n_preds = np.sum(outputs >= std * thresh)
        return np.abs(n_preds - target * song_len)
    res = optimize.minimize_scalar(fn, bounds=bounds)
    print("Threshold:", res.x)
    return res.x

def smooth_outputs(outputs, q=10):
    s = pd.Series(outputs)
    avg = s.rolling(q).median()
    avg = avg.fillna(avg[q+1])
    smooth_outputs = (outputs - avg).values

    # Roll backwards to remove lag.
    smooth_outputs = np.roll(smooth_outputs, int(q/2))
    return smooth_outputs

def edit_mismatched_holds(preds, outputs):
    """
    Currently unused/in testing.
    """

    # Ignore the existance of type 3/4s.
    preds = np.maximum(preds, 2)

    is_held = np.zeros(4).astype(bool)
    hold_start = np.zeros(4).astype(int)

    count = 0
    for i in range(len(preds)):
        row = preds[i]
        hold_idxs = np.where(row == 2)[0]

        # What to do if found a bad end to the match.
        bad_inds = np.logical_and(row != 2, is_held)

        if bad_inds.sum() > 0:
            # Either change step into hold, or make both steps.
            # Random sample for decision lmao.
            step_score = np.exp(outputs[i, :, 1])
            hold_score = np.exp(outputs[i, :, 1:]).sum(axis=1)
            probs = step_score / hold_score

            new_steps = np.random.binomial(1, 1 - probs) + 1
            preds[i, bad_inds] = new_steps[bad_inds]

            # If turning into steps, change where the hold
            # should have started into a step.
            update_steps = np.logical_and(new_steps == 1, bad_inds)
            update_idxs = np.where(update_steps)[0]
            preds[hold_start[update_steps], update_idxs] = 1

            # No matter which event, only 1 change.
            count += np.sum(new_steps == 1)

            # Either ends a hold, or is a step.
            is_held[bad_inds] = False

        # Completely valid holds. Toggle status.
        hold_start_inds = np.logical_and(row == 2, ~bad_inds)
        is_held[hold_start_inds] = ~is_held[hold_start_inds]
        hold_start[hold_start_inds] = i

    print(f"Modified {count} hold notes.")

    return preds



def get_steps(step_tensor):
    """
    Converts outputs into steps.
    Also fills in "blank" steps.
    """

    step = np.argmax(step_tensor, axis=2)

    blank_step_idxs = step.sum(axis=1) == 0

    print("Blank steps: ", 1 - np.mean(blank_step_idxs))

    # Sometimes no steps will be predicted.
    # If this happens, take the largest value instead.
    # Drop "no step" column.
    blank_step_tensor = step_tensor[blank_step_idxs, :, 1:].reshape((-1, 16))
    n_blanks = blank_step_tensor.shape[0]
    blank_step_type_idx = blank_step_tensor.argmax(axis=1)
    step[blank_step_idxs, blank_step_type_idx // 4] = \
            1 + blank_step_type_idx % 4


    return step



def reduce_jumps(output_tensor, alpha=1, prior=1.13):
    """
    Not used, since removing doubles seems to do it well enough.
    """

    def exp_jumps(output_tensor, T):
        log_prob_step = output_tensor[:,:,0] + T - np.log(
                np.exp(output_tensor[:,:,0] + T) +
                np.sum(np.exp(output_tensor[:,:,1:]), axis=2))

        prob_nostep = 1 - np.exp(log_prob_step)

        return np.mean(np.sum(prob_nostep, axis=1))

    data_exp_jumps = exp_jumps(output_tensor, 0)

    def fn(T):
        exp_jumps_T = exp_jumps(output_tensor, T)

        prior_loss = np.mean((exp_jumps_T - prior)**2)
        data_loss = np.mean((exp_jumps_T - data_exp_jumps)**2)

        return prior_loss + alpha * data_loss

    print("Data expectation:", data_exp_jumps)

    res = optimize.minimize(fn, 0)

    output_tensor[:,:,0] += res.x
    print("Final exp jumps:", exp_jumps(output_tensor, 0))
    return output_tensor


def filter_triples(output_tensor, steps):
    n_row_steps = (steps > 0).sum(axis=1)

    triple_steps = n_row_steps >= 3
    n_triple_steps = triple_steps.sum()
    print(f"Fixing {n_triple_steps} rows of triple+ steps")
    triple_steps_idx = np.where(triple_steps)[0]

    triples_output = output_tensor[triple_steps, :, :]

    # n_triple_steps x 2
    new_idxs = np.argsort(triples_output.max(axis=2))[:, -2:]

    for i in range(n_triple_steps):
        new_steps = np.zeros(4)
        new_step_idxs = new_idxs[i, :]
        new_step_types = np.argmax(triples_output[i, new_step_idxs, 1:], axis=1) + 1
        new_steps[new_step_idxs] = new_step_types
        steps[triple_steps_idx[i],:] = new_steps


    return steps


def remove_doubles(output_tensor, step, beats_before, bpm, ms=128):
    """
    ms: Minimum time between double notes.
    """
    beats_before = beats_before.reshape(-1)
    # if not (output_tensor.shape[0] == step.shape[0] == beats_before.shape[0]):
        # print("ERROR, bad shapes.")
        # print(output_tensor.shape)
        # print(step.shape)
        # print(beats_before.shape)

    assert output_tensor.shape[0] == step.shape[0] == beats_before.shape[0]

    count = 0
    time_before = beats_before * (60 / bpm)
    secs = ms / 1000

    for i in range(1, len(output_tensor)):
        c_step = step[i]
        p_step = step[i-1]
        output = output_tensor[i, :, :]

        # Number of steps to look back.
        cand_steps = time_before[max(0, i-10):i]
        prev_times = np.cumsum(cand_steps[::-1])
        n_prev_steps = np.sum(prev_times < secs)
        if n_prev_steps==0:
            continue


        # Positions of prior steps within ms.
        history = step[(i - n_prev_steps):i,:]
        valid_ind = history.sum(axis=0) == 0
        valid_idxs = np.where(valid_ind)[0]


        # If no viable alternative, drop the note.
        if len(valid_idxs) == 0:
            count += 1
            step[i,:] = np.zeros(4)
            continue

        # Index of current steps.
        idx = np.where(c_step > 0)[0]

        # Skip if no conflict.
        if len(np.intersect1d(valid_idxs, idx)) == len(idx):
            continue

        # If replacing, pick next most probable valid step.
        # Replace with normal note.
        step_type = 1
        alt_probs = output[valid_ind, step_type]
        alt_idx = valid_idxs[np.argmax(alt_probs)]

        res = np.zeros(4)
        res[alt_idx] = step_type

        step[i,:] = res
        count += 1

    print(f"Modified {count} rows.")

    return step


