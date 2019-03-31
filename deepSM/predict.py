from deepSM import StepPlacement
from deepSM import SMDataset
from deepSM import wavutils

import torch
import torch.utils.data as datautils


def get_fft_from_wav(wavdata):
    wav_frames = len(wavdata) // 512 + 1
    front_pad_frames, padded_wav = \
            wavutils.pad_wav(0, wav_frames, wavdata)

    fft_features = wavutils.gen_fft_features(padded_wav, log=True)
    return fft_features


model_dir = '/home/lence/dev/deepStep/models/'
placement_model = model_dir + '/fraxtil_log_rnn_epoch_1_2019-03-28_01-27-54.sd'
def predict_step_placements(wavf, diff, model_name=placement_model):
    rate, wavdata = wavutils.read_wav(wavf)

    fft_features = get_fft_from_wav(wavdata)

    smd = SMDataset.SMDataset(
            'WAVGen',
            [diff],
            fft_features,
            chunk_size=-1)

    loader = datautils.DataLoader(smd)

    placement_model = StepPlacement.RegularizedRecurrentStepPlacementModel()
    placement_model.load_state_dict(torch.load(model_name))
    placement_model.cuda()
    placement_model.eval()

    d = next(iter(loader))
    batch_fft = d['fft_features'].cuda()
    batch_diffs = d['diff'].cuda()

    with torch.no_grad():
        output = placement_model(batch_fft, batch_diffs)

    output = torch.sigmoid(output)

    return output
