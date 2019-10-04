import os
import argparse
import numpy as np
import json

import torch
import torch.utils.data as datautils

from deepSM import post_processing
from deepSM import SMDUtils
from deepSM import SMDataset
from deepSM import SMGenDataset
from deepSM import generate_sm_file
from deepSM import beat_alignment
from deepSM import StepPlacement
from deepSM import StepGeneration
from deepSM import bpm_estimator
from deepSM import wavutils
from deepSM import utils
from deepSM import beat_time_converter
from deepSM import convert_to_gen_dataset

prior = 1.18
threshold = 0.01

def SMPipeline(
        song_file,
        placement_model_dir,
        gen_model_dir,
        prior=prior,
        threshold = threshold,
        smooth=False,
        drop_subdivs=None,
        log=False,
        bpm=None,
        use_cuda=True):

    print(f"Processing .wav file {song_file.split('/')[-1]}")
    print(f"Using placement model {placement_model_dir.split('/')[-1]}")
    print(f"Using gen model {gen_model_dir.split('/')[-1]}")
    print("CUDA:", use_cuda)
    print("Prior:", prior)
    print("Threshold:", threshold)
    diff_names = ['Challenge', 'Hard', 'Medium', 'Easy', 'Beginner']
    # diff_names = ['Challenge', 'Hard']


    try:
        threshold = float(threshold)
        threshold_dict = False
    except:
        with open(threshold) as f:
            threshold_dict = json.load(f)


    song_name = song_file.split('/')[-1].split('.')[0]

    # Prepare original wav file for feature engineering.
    rate, wavdata = wavutils.read_wav(song_file)
    wav_frames = len(wavdata) // 512 + 1
    front_pad_frames, padded_wav = \
            wavutils.pad_wav(0, wav_frames, wavdata)

    # Generate audio features.
    # Contains padding.
    fft_features = wavutils.gen_fft_features(padded_wav, log=log)
    n_frames = fft_features.shape[1]

    # Predict step placements.
    placement_model = StepPlacement.RegularizedRecurrentStepPlacementModel(log=True)
    placement_model.load_state_dict(torch.load(placement_model_dir))
    if use_cuda is not False and use_cuda is not None:
        if isinstance(use_cuda, int):
            placement_model.cuda()
        else:
            placement_model.cuda()
    placement_model.eval()

    preds = {}
    smds = {}
    frame_idxs = {}
    outputs = {}
    for diff in diff_names:
        diff_code = utils.difficulties[diff]

        smd = SMDataset.SMDataset(
            song_name,
            [diff],
            fft_features,
            chunk_size=-1)
        smds[diff] = smd

        loader = datautils.DataLoader(smd)

        d = next(iter(loader))
        # batch_fft = d['fft_features'].cuda()
        # batch_diffs = d['diff'].cuda()

        batch_fft, batch_diffs = placement_model.prepare_data(d, use_labels=False)

        with torch.no_grad():
            output = placement_model(batch_fft, batch_diffs)


        if smooth:
            smoothed = post_processing.smooth_outputs(output[0,:,0].cpu().numpy())
            if threshold_dict:
                threshold = threshold_dict[diff]
            pred = (smoothed > np.std(smoothed) * threshold)
            print('std:', np.std(smoothed), 'p pos:', np.sum(pred))
        else:
            output = torch.sigmoid(output).reshape(-1)
            pred = (output > threshold).cpu().numpy()

        frame_idx = np.where(pred == 1)[0]

        outputs[diff] = output
        preds[diff] = pred
        frame_idxs[diff] = frame_idx

    print("Steps placed.")
    # Hopefully free up some CUDA memory.
    del placement_model

    if np.sum(preds['Hard']) < 100:
        nsteps = np.sum(preds['Hard'])
        print(f"Warning: Too few steps {nsteps}. Something probably broke.")
        return


    if bpm is None:
        bpm = bpm_estimator.refined_bpm_estimate(preds)

    offset, _ = beat_alignment.frames_to_measures(preds['Hard'], bpm)


    print("Final BPM:", bpm)

    print("Generating steps.")
    # gen_model = StepGeneration.RegularizedRecurrentStepGenerationModel(log=False)
    gen_model = StepGeneration.RegularizedRecurrentStepGenerationModel(log=True)
    gen_model.load_state_dict(torch.load(gen_model_dir))
    if use_cuda:
        if isinstance(use_cuda, int):
            gen_model.cuda()
        else:
            gen_model.cuda()
    gen_model.eval()

    diff_divnotes = []
    diff_steps = []
    for diff in diff_names:
        smd = smds[diff]

        diff_order, diff_features = \
                convert_to_gen_dataset.get_generation_features(
                        smd, bpm, [frame_idxs[diff]], use_labels=False)

        smgd = SMGenDataset.SMGenDataset(
            song_name,
            [diff],
            diff_features,
            chunk_size=-1)

        loader = datautils.DataLoader(smgd)
        d = next(iter(loader))

        batch_fft, batch_diff, batch_beats_before, batch_beats_after = \
               gen_model.prepare_data(d, use_labels=False)

        with torch.no_grad():
            step_outputs = gen_model(
                    batch_fft,
                    batch_diff,
                    batch_beats_before,
                    batch_beats_after)

        step_outputs = step_outputs.cpu().numpy()
        step_tensor = step_outputs.reshape((-1, 4, 5))

        # step_tensor = post_processing.reduce_jumps(step_tensor, prior=prior)

        step = post_processing.get_steps(step_tensor)
        assert np.sum(step.sum(axis=1) > 0) == step.shape[0], 'get steps'
        step = post_processing.filter_triples(step_tensor, step)
        assert np.sum(step.sum(axis=1) > 0) == step.shape[0], 'filter triples'


        _, divnotes = beat_alignment.frames_to_measures(
                preds[diff], bpm, offset=offset, drop_subdivs=drop_subdivs)

        btc = beat_time_converter.BeatTimeConverter(offset, [(0, bpm)], [])

        # Filter notes that are too close based on aligned times.
        # Get aligned times.
        beat_idxs = []
        for m_idx in range(len(divnotes[0])):
            measure = np.array(divnotes[1][m_idx])
            divs = divnotes[0][m_idx]
            beat = m_idx * 4 + 4 * measure / divs
            beat_idxs.extend(list(beat))

        aligned_times = btc.beat_to_time(beat_idxs)
        aligned_diffs = np.r_[12 * 60 / bpm, aligned_times[1:] - aligned_times[:-1]]

        if not (step_tensor.shape[0] == step.shape[0] == aligned_diffs.shape[0]):
            print("ERROR, bad shapes.")
            print(step_tensor.shape)
            print(step.shape)
            print(aligned_diffs.shape)
            print(len(divnotes[0]))
            print(sum(map(len, divnotes[1])))
            print(len(beat_idxs))
            print(len(aligned_times))

        step = post_processing.remove_doubles(step_tensor, step,
                aligned_diffs, bpm)

        # step = post_processing.edit_mismatched_holds(step, step_tensor)
        print("Hold counts:", np.mean(step > 1))

        print("N steps:", np.sum(step.sum(axis=1) > 0))


        diff_divnotes.append(divnotes)
        diff_steps.append(step)

    print("Steps generated. Outputting .sm file.")


    generate_sm_file.to_SMFile(song_name, song_name + '.wav', diff_names,
            offset, bpm,
            diff_divnotes, diff_steps, subtitle=utils.timestamp(),
            sm_path=f'{song_file[:-4]}.sm',
            comment=f"{placement_model_dir}, {gen_model_dir}")



if __name__ == '__main__':


    parser = argparse.ArgumentParser(
            description="Generate step maps from raw a .wav file.")

    parser.add_argument("song_file", type=str,
            help="Path to directory containing a single channel .wav file.")
    parser.add_argument("placement_model", type=str,
            help="Path to the placement model weights.")
    parser.add_argument("gen_model", type=str,
            help="Path to the step generation model weights.")

    parser.add_argument("threshold",
            help="""Integer or string value (Typically around 1.7 for smoothed) for the threshold.
If string, should point to a threshold dictionary.
""")

    parser.add_argument("--bpm", type=int)
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--prior", type=float, default=prior)

    # Drop subdivs by default, and use smoothing by default.
    # parser.add_argument("--drop_subdivs", type=int, default=None)
    parser.add_argument("--drop_subdivs", action='store_false')
    parser.add_argument("--smooth", action='store_false')

    args = parser.parse_args()

    SMPipeline(
            args.song_file,
            args.placement_model,
            args.gen_model,
            args.prior,
            args.thresh,
            args.smooth,
            args.drop_subdivs,
            log=False,
            bpm=args.bpm,
            use_cuda=not args.cpu)
