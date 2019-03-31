import generate_sm
import os
import argparse
from concurrent import futures

base_dir = os.getcwd()
song_dirs = os.listdir()

model_dir = '/home/lence/dev/deepStep/models/'
placement_model = model_dir + '/fraxtil_log_rnn_epoch_1_2019-03-28_01-27-54.sd'
# gen_model = model_dir + '/RegularizedRecurrentStepGenerationModel_2019-03-25_23-03-26.sd'
gen_model = model_dir + '/fraxtil_log_rnn_gen_2019-03-29_11-47-48.sd'

def get_bpm(fname):
    with open(fname) as f:
        bpmline = next(filter(lambda x: x.startswith('#BPM'), f.readlines()))
        bpm = bpmline.split(':')[-1][:-2].split('=')[1]
        return float(bpm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_bpm', action='store_true')
    parser.add_argument('--placement_model', type=str,
            default=placement_model)
    parser.add_argument('--gen_model', type=str,
            default=gen_model)
    parser.add_argument('--n_threads', type=int, default=-1)
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()

    placement_model = os.path.abspath(args.placement_model)
    gen_model = os.path.abspath(args.gen_model)

    wavfs = []
    bpms = []

    for song_dir in song_dirs:
        os.chdir(base_dir + '/' + song_dir)

        files = os.listdir()
        wavf = next(filter(lambda x: x.endswith('wav'), files))
        smf = next(filter(lambda x: x.endswith('sm'), files))

        if args.infer_bpm:
            bpm = get_bpm(smf)
        else:
            bpm = None

        wavfs.append(os.path.abspath(wavf))
        bpms.append(bpm)

    os.chdir(base_dir)


    if args.n_threads > 0:
        n = len(song_dir)
        with futures.ProcessPoolExecutor(max_workers=args.n_threads) as executor:
            res = executor.map(
                    generate_sm.SMPipeline,
                    wavfs,
                    [placement_model] * n,
                    [gen_model] * n, [False] * n,
                    bpms,
                    [not args.cpu] * n)

            for r in res:
                print("RES")
                print(r)
    else:
        for wavf, bpm in zip(wavfs, bpms):
            generate_sm.SMPipeline(
                wavf,
                placement_model,
                gen_model,
                log=False,
                bpm=bpm,
                use_cuda=not args.cpu)

