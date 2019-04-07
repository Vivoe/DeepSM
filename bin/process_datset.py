import generate_sm
import os
import argparse
from concurrent import futures
import numpy as np

base_dir = os.getcwd()
song_dirs = os.listdir()


def get_bpm(fname):
    with open(fname) as f:
        bpmline = next(filter(lambda x: x.startswith('#BPM'), f.readlines()))
        bpm = bpmline.split(':')[-1][:-2].split('=')[1]
        return float(bpm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('placement_model', type=str)
    parser.add_argument('gen_model', type=str)
    parser.add_argument("threshold",
            help="""Integer or string value (Typically around 1.7 for smoothed) for the threshold.
If string, should point to a threshold dictionary.
""")

    parser.add_argument('--infer_bpm', action='store_true')
    parser.add_argument('--n_threads', type=int, default=-1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument("--prior", type=float, default=generate_sm.prior,
            help="Currently unused.")
    parser.add_argument("--drop_subdivs", action='store_false',
            help="Snap notes to at most 24th subdivisions. Default: True")

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
                    [args.threshold] * n,
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
                args.threshold,
                True,
                args.drop_subdivs,
                log=False,
                bpm=bpm,
                use_cuda=not args.cpu)

