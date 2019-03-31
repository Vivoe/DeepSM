import os
import argparse
import numpy as np

import torch
import torch.utils.data as datautils

import convert_to_gen_dataset

from deepSM import post_processing
from deepSM import SMDUtils
from deepSM import SMDataset
from deepSM import SMGenDataset
from deepSM import generate_sm_file
from deepSM import StepPlacement
from deepSM import StepGeneration
from deepSM import bpm_estimator
from deepSM import wavutils
from deepSM import utils

# from memory_profiler import profile
# @profile
def SMPipeline(
        song_file,
        placement_model_dir,
        gen_model_dir,
        log=False,
        bpm=None,
        use_cuda=True):

    print(f"Processing .wav file {song_file.split('/')[-1]}")
    print(f"Using placement model {placement_model_dir.split('/')[-1]}")
    print(f"Using gen model {gen_model_dir.split('/')[-1]}")
    print("Log", log)
    print("Cuda", use_cuda)
    diff_names = ['Challenge', 'Hard', 'Medium', 'Easy', 'Beginner']

    # Threshold computed off of training data.
    threshold = 0.1

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
    if use_cuda:
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

        output = torch.sigmoid(output)

        pred = (output[0,:,0] > threshold).cpu().numpy()
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
        # Search for best BPM from candidates.
        # Consider also searching off-by-one BPMs.
        # bpms = []
        # for diff in diff_names:
            # bpm = bpm_estimator.est_bpm(preds[diff])
            # bpms.append(bpm)

        # candidate_bpms = []
        # for bpm in bpms:
            # candidate_bpms.append(bpm-1)
            # candidate_bpms.append(bpm)
            # candidate_bpms.append(bpm+1)

        # candidate_bpms = list(set(candidate_bpms))
        # print("Candidate BPMS:", candidate_bpms)

        # bpm_scores = []
        # for bpm in candidate_bpms:
            # print("Processing", bpm)
            # score = 0
            # for diff in diff_names:
                # print("Diff", diff)

                # offset, divnotes = \
                     # generate_sm_file.frames_to_measures(preds[diff], bpm)

                # score += sum(divnotes[0])

            # bpm_scores.append(score)

        # bpm_idx = np.argmin(bpm_scores)
        # bpm = candidate_bpms[bpm_idx]
        # print("BPM scores:", bpm_scores)

    offset, _ = generate_sm_file.frames_to_measures(preds['Hard'], bpm)


    print("Final BPM:", bpm)

    print("Generating steps.")
    # gen_model = StepGeneration.RegularizedRecurrentStepGenerationModel(log=False)
    gen_model = StepGeneration.RegularizedRecurrentStepGenerationModel(log=True)
    gen_model.load_state_dict(torch.load(gen_model_dir))
    if use_cuda:
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
        # batch_fft = d['fft_features'].float().cuda()
        # batch_fft = 1e-4 + 10**batch_fft
        # batch_beats_before = d['beats_before'].float().unsqueeze(2).cuda()
        # batch_beats_after = d['beats_after'].float().unsqueeze(2).cuda()
        # batch_diff = d['diff'].float().cuda()

        batch_fft, batch_beats_before, batch_beats_after, batch_diff = \
               gen_model.prepare_data(d, use_labels=False)
        # batch_fft = 1e-4 + 10**batch_fft

        with torch.no_grad():
            step_outputs = gen_model(
                    batch_fft,
                    batch_diff,
                    batch_beats_before,
                    batch_beats_after)

        step_outputs = step_outputs.cpu().numpy()
        step_tensor = step_outputs.reshape((-1, 4, 5))

        step = post_processing.get_steps(step_tensor)
        step = post_processing.remove_doubles(step_tensor, step,
                batch_beats_before.cpu().numpy(), bpm)


        _, divnotes = generate_sm_file.frames_to_measures(
                preds[diff], bpm, offset=offset)

        diff_divnotes.append(divnotes)
        diff_steps.append(step)

    print("Steps generated. Outputting .sm file.")


    generate_sm_file.to_SMFile(song_name, song_name + '.wav', diff_names,
            offset, bpm,
            diff_divnotes, diff_steps, subtitle=utils.timestamp(),
            sm_path=f'{song_file[:-4]}.sm')


if __name__ == '__main__':

    model_dir = '/home/lence/dev/deepStep/models/'
    placement_model = model_dir + '/fraxtil_log_rnn_epoch_1_2019-03-28_01-27-54.sd'
    gen_model = model_dir + '/fraxtil_log_rnn_gen_2019-03-29_11-47-48.sd'


    parser = argparse.ArgumentParser(
            description="Generate step maps from raw a .wav file.")

    parser.add_argument("song_file", type=str,
            help="Path to directory containing a single channel .wav file.")
    parser.add_argument("--placement_model", type=str, default=placement_model,
            help="Path to the placement model weights.")
    parser.add_argument("--gen_model", type=str, default=gen_model,
            help="Path to the step generation model weights.")
    parser.add_argument("--bpm", type=int)
    parser.add_argument("--cpu", action='store_true')

    args = parser.parse_args()

    SMPipeline(
            args.song_file,
            args.placement_model,
            args.gen_model,
            log=False,
            bpm=args.bpm,
            use_cuda=not args.cpu)
