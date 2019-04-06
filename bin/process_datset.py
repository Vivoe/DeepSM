import generate_sm
import os
import argparse
from concurrent import futures
import numpy as np

base_dir = os.getcwd()
song_dirs = os.listdir()

# model_dir = '/home/lence/dev/deepStep/models/'
# placement_model = model_dir + '/fraxtil_log_rnn_epoch_1_2019-03-28_01-27-54.sd'
# gen_model = model_dir + '/RegularizedRecurrentStepGenerationModel_2019-03-25_23-03-26.sd'
# gen_model = model_dir + '/fraxtil_log_rnn_gen_2019-03-29_11-47-48.sd'

# placement_model = model_dir + '/jubo_log_rnn_epoch_1_2019-03-30_17-06-40.sd'
# gen_model = model_dir + '/jubo_log_rnn_gen_2019-03-30_20-38-24.sd'
# gen_model = model + '/jubo_log_rnn_gen_2019-03-31_14-12-13.sd'



def get_bpm(fname):
    with open(fname) as f:
        bpmline = next(filter(lambda x: x.startswith('#BPM'), f.readlines()))
        bpm = bpmline.split(':')[-1][:-2].split('=')[1]
        return float(bpm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_bpm', action='store_true')
    parser.add_argument('--placement_model', type=str,
            default=generate_sm.placement_model)
    parser.add_argument('--gen_model', type=str,
            default=generate_sm.gen_model)
    parser.add_argument('--n_threads', type=int, default=-1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument("--prior", type=float, default=generate_sm.prior)
    parser.add_argument("--drop_subdivs", type=int, default=1)
    parser.add_argument("--thresh", default=1.65)

    args = parser.parse_args()

    placement_model = os.path.abspath(args.placement_model)
    gen_model = os.path.abspath(args.gen_model)

    wavfs = []
    bpms = []

    for song_dir in song_dirs:
        os.chdir(base_dir + '/' + song_dir)

        files = os.listdir()
        wavf = next(filter(lambda x: x.endswith('wav'), files))

        if args.infer_bpm:
            smf = next(filter(lambda x: x.endswith('sm'), files))
            bpm = get_bpm(smf)
        else:
            bpm = None

        wavfs.append(os.path.abspath(wavf))
        bpms.append(bpm)

    os.chdir(base_dir)


    n = len(song_dir)
    if args.n_threads > 0:
        with futures.ProcessPoolExecutor(max_workers=args.n_threads) as executor:
            res = executor.map(
                    generate_sm.SMPipeline,
                    wavfs,
                    [placement_model] * n,
                    [gen_model] * n,
                    [args.prior] * n,
                    [args.thresh] * n,
                    [True] * n,
                    [args.drop_subdivs] * n,
                    [False] * n,
                    bpms,
                    np.arange(n) % 8)

            for r in res:
                print("RES")
                print(r)
    else:
        for i, (wavf, bpm) in enumerate(zip(wavfs, bpms)):
            print(f"Computing sm file {i}/{n}")
            generate_sm.SMPipeline(
                wavf,
                placement_model,
                gen_model,
                args.prior,
                args.thresh,
                args.smooth,
                args.drop_subdivs,
                log=False,
                bpm=bpm,
                use_cuda=not args.cpu)

