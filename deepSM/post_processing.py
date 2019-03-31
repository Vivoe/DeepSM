import numpy as np

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

def remove_doubles(output_tensor, step, beats_before, bpm, ms=125):
    """
    ms: Minimum time between double notes.
    """
    beats_before = beats_before[0, :, 0]
    assert output_tensor.shape[0] == step.shape[0] == beats_before.shape[0]

    count = 0

    for i in range(1, len(output_tensor)):
        c_step = step[i]
        p_step = step[i-1]
        output = output_tensor[i, :, :]
        time_before = beats_before[i] * 60 / bpm
        secs = ms / 1000

        # Index of steps.
        idx = np.where(c_step > 0)[0]

        valid_idxs = np.where(p_step == 0)[0]

        # Only process if within X ms of the previous note.
        if time_before >= secs:
            continue

        # Only process if current step has only 1 note.
        if len(idx) > 1:
            continue

        # Skip if no viable alternative.
        if len(valid_idxs) == 0:
            continue


        idx = idx[0]
        step_type = c_step[idx]

        # Skip if no conflict.
        if p_step[idx] == 0:
            continue


        # If replacing, pick next most probable valid step.

        alt_probs = output[p_step == 0, step_type]
        alt_idx = valid_idxs[np.argmax(alt_probs)]

        res = np.zeros(4)
        res[alt_idx] = step_type

        step[i] = res

        count += 1

    print(f"Modified {count} rows.")

    return step


