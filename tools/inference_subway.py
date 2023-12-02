if __name__ == '__main__':
    import os
     
    gpu_use = "0"

    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import soundfile as sf
from demucs.states import load_model
from demucs import pretrained
from demucs.apply import apply_model
import onnxruntime as ort
from time import time
import librosa
import hashlib
from scipy import signal
import gc
import yaml
from ml_collections import ConfigDict
import sys
import math
import pathlib
import warnings
from tools.tfc_tdf_v3 import TFC_TDF_net
from scipy.signal import resample_poly
import datetime
import os
import uuid
class Conv_TDF_net_trim_model(nn.Module):
    def __init__(self, device, target_name, L, n_fft, hop=1024):

        super(Conv_TDF_net_trim_model, self).__init__()

        self.dim_c = 4
        self.dim_f, self.dim_t = 3072, 256
        self.n_fft = n_fft
        self.hop = hop
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(device)
        self.target_name = target_name

        out_c = self.dim_c * 4 if target_name == '*' else self.dim_c
        self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t]).to(device)

        self.n = L // 2

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, self.dim_c, self.n_bins, self.dim_t])
        return x[:, :, :self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = self.freq_pad.repeat([x.shape[0], 1, 1, 1]) if freq_pad is None else freq_pad
        x = torch.cat([x, freq_pad], -2)
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 2, self.n_bins, self.dim_t])
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        return x.reshape([-1, 2, self.chunk_size])

    def forward(self, x):
        x = self.first_conv(x)
        x = x.transpose(-1, -2)

        ds_outputs = []
        for i in range(self.n):
            x = self.ds_dense[i](x)
            ds_outputs.append(x)
            x = self.ds[i](x)

        x = self.mid_dense(x)
        for i in range(self.n):
            x = self.us[i](x)
            x *= ds_outputs[-i - 1]
            x = self.us_dense[i](x)

        x = x.transpose(-1, -2)
        x = self.final_conv(x)
        return x


def get_models(name, device, load=True, vocals_model_type=0):
    if vocals_model_type == 2:
        model_vocals = Conv_TDF_net_trim_model(
            device=device,
            target_name='vocals',
            L=11,
            n_fft=7680
        )
    elif vocals_model_type == 3:
        model_vocals = Conv_TDF_net_trim_model(
            device=device,
            target_name='vocals',
            L=11,
            n_fft=6144
        )

    return [model_vocals]

def demix_base_mdxv3(config, model, mix, device):
        N = options["overlap_MDXv3"]
        mix = torch.tensor(mix, dtype=torch.float32)
        try:
            S = model.num_target_instruments
        except Exception as e:
            S = model.module.num_target_instruments

        mdx_window_size = config.inference.dim_t
        
        # batch_size = config.inference.batch_size
        batch_size = 1
        C = config.audio.hop_length * (mdx_window_size - 1)
        

        H = C // N
        L = mix.shape[1]
        pad_size = H - (L - C) % H
        mix = torch.cat([torch.zeros(2, C - H), mix, torch.zeros(2, pad_size + C - H)], 1)
        mix = mix.to(device)

        chunks = []
        i = 0
        while i + C <= mix.shape[1]:
            chunks.append(mix[:, i:i + C])
            i += H
        chunks = torch.stack(chunks)

        batches = []
        i = 0
        while i < len(chunks):
            batches.append(chunks[i:i + batch_size])
            i = i + batch_size

        X = torch.zeros(S, 2, C - H) if S > 1 else torch.zeros(2, C - H)
        X = X.to(device)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                for batch in batches:
                    # self.running_inference_progress_bar(len(batches))
                    x = model(batch)
                    for w in x:
                        a = X[..., :-(C - H)]
                        b = X[..., -(C - H):] + w[..., :(C - H)]
                        c = w[..., (C - H):]
                        X = torch.cat([a, b, c], -1)

        estimated_sources = X[..., C - H:-(pad_size + C - H)] / N
        
        if S > 1:
            return {k: v for k, v in zip(config.training.instruments, estimated_sources.cpu().numpy())}
        else:
            est_s = estimated_sources.cpu().numpy()
            return est_s
def print_memory_usage(message):
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)    # 指定GPU的id
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    now = datetime.datetime.now()  # 获取当前时间
    print(f"当前时间: {now.strftime('%H:%M:%S')}")
    print(f"{message}-->当前分配显存: {int(meminfo.used / 1024**2)} MiB\n")
def demix_full_mdx23c(mix, device):

    # model_folder = os.path.dirname(os.path.abspath(__file__)) + '/models/'
    model_folder = '/root/Pretrained_ model/MDX23_models/'
    remote_url_mdxv3 = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/MDX23C-8KFFT-InstVoc_HQ.ckpt'
    remote_url_conf = 'https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_2_stem_full_band_8k.yaml'
    if not os.path.isfile(model_folder+'MDX23C-8KFFT-InstVoc_HQ.ckpt'):
        torch.hub.download_url_to_file(remote_url_mdxv3, model_folder+'MDX23C-8KFFT-InstVoc_HQ.ckpt')
    if not os.path.isfile(model_folder+'model_2_stem_full_band_8k.yaml'):
        torch.hub.download_url_to_file(remote_url_conf, model_folder+'model_2_stem_full_band_8k.yaml')


    with open(model_folder + 'model_2_stem_full_band_8k.yaml') as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    model = TFC_TDF_net(config)
    model.load_state_dict(torch.load(model_folder+'MDX23C-8KFFT-InstVoc_HQ.ckpt'))
    device = torch.device(device)

    model = model.to(device)
    model.eval()
        
    sources = demix_base_mdxv3(config, model, mix, device)

    model = model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return sources


def demix_base(mix, device, models, infer_session,first_flag):

    lock = options["lock"]
    inference_input_queue = options["inference_input_queue"]
    output_queue = options["output_queue"]
    worker_id = options["worker_id"]
    start_time = time()
    sources = []
    n_sample = mix.shape[1]
    for model in models:
        trim = model.n_fft // 2
        gen_size = model.chunk_size - 2 * trim
        pad = gen_size - n_sample % gen_size
        mix_p = np.concatenate(
            (
                np.zeros((2, trim)),
                mix,
                np.zeros((2, pad)),
                np.zeros((2, trim))
            ), 1
        )

        mix_waves = []
        i = 0
        while i < n_sample + pad:
            waves = np.array(mix_p[:, i:i + model.chunk_size])
            mix_waves.append(waves)
            i += gen_size
        mix_waves = np.array(mix_waves)
        mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(device)
        with torch.no_grad():
            _ort = infer_session
            stft_res = model.stft(mix_waves)   
            
            task_id = str(uuid.uuid4())  # 生成唯一的任务ID
            inference_input_queue.put({'task_id': task_id, 'input_data': stft_res.cpu().numpy(), 'worker_id': worker_id,"infer_session_id":infer_session})
            result = output_queue.get()
            assert result['task_id'] == task_id, f"Task ID mismatch for worker {worker_id}"
            res = result['result']
            
            ten = torch.tensor(res)
            tar_waves = model.istft(ten.to(device))
            tar_waves = tar_waves.cpu()
            tar_signal = tar_waves[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).numpy()[:, :-pad]

        sources.append(tar_signal)
    # print('Time demix base: {:.2f} sec'.format(time() - start_time))
    return np.array(sources)

def demix_full(mix, device, chunk_size, models, infer_session, overlap=0.2, bigshifts=1):
    first_flag = 1
    start_time = time()
    step = int(chunk_size * (1 - overlap))
    shift_number = bigshifts # must not be <= 0 !
    if shift_number < 1:
        shift_number=1
    
    mix_length = mix.shape[1] / 44100
    if shift_number > int(mix_length):
        shift_number = int(mix_length - 1)
    shifts = [x for x in range(shift_number)]
    results = []
    for shift in shifts:
        shift_samples = int(shift * 44100)
        # print(f"shift_samples = {shift_samples}")
        shifted_mix = np.concatenate((mix[:, -shift_samples:], mix[:, :-shift_samples]), axis=-1)
        # print(f"shifted_mix shape = {shifted_mix.shape}")
        result = np.zeros((1, 2, shifted_mix.shape[-1]), dtype=np.float32)
        divider = np.zeros((1, 2, shifted_mix.shape[-1]), dtype=np.float32)
        total = 0
        for i in range(0, shifted_mix.shape[-1], step):
            total += 1
            start = i
            end = min(i + chunk_size, shifted_mix.shape[-1])
            mix_part = shifted_mix[:, start:end]
            # print(f"mix_part shape = {mix_part.shape}")
            sources = demix_base(mix_part, device, models, infer_session,first_flag)
            first_flag +=1
            result[..., start:end] += sources
            divider[..., start:end] += 1
        result /= divider
        # print(f"result shape = {result.shape}")
        result = np.concatenate((result[..., shift_samples:], result[..., :shift_samples]), axis=-1)
        results.append(result)
    results = np.mean(results, axis=0)
    return results


class EnsembleDemucsMDXMusicSeparationModel:
    """
    Doesn't do any separation just passes the input back as output
    """
    def __init__(self, options):
        """
            options - user options
        """
        # print(options)

        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        if 'cpu' in options:
            if options['cpu']:
                device = 'cpu'
        # print('Use device: {}'.format(device))
        self.single_onnx = False
        if 'single_onnx' in options:
            if options['single_onnx']:
                self.single_onnx = True
                # print('Use single vocal ONNX')
        self.overlap_demucs = float(options['overlap_demucs'])
        self.overlap_MDX = float(options['overlap_MDX'])
        if self.overlap_demucs > 0.99:
            self.overlap_demucs = 0.99
        if self.overlap_demucs < 0.0:
            self.overlap_demucs = 0.0
        if self.overlap_MDX > 0.99:
            self.overlap_MDX = 0.99
        if self.overlap_MDX < 0.0:
            self.overlap_MDX = 0.0
        model_folder = '/root/Pretrained_ model/MDX23_models/'
        """
        
        remote_url = 'https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/04573f0d-f3cf25b2.th'
        model_path = model_folder + '04573f0d-f3cf25b2.th'
        if not os.path.isfile(model_path):
            torch.hub.download_url_to_file(remote_url, model_folder + '04573f0d-f3cf25b2.th')
        model_vocals = load_model(model_path)
        model_vocals.to(device)
        self.model_vocals_only = model_vocals
        """
        if options['vocals_only'] is False:
            self.models = []
            self.weights_vocals = np.array([10, 1, 8, 9])
            self.weights_bass = np.array([19, 4, 5, 8])
            self.weights_drums = np.array([18, 2, 4, 9])
            self.weights_other = np.array([14, 2, 5, 10])
    
            model1 = pretrained.get_model('htdemucs_ft')
            model1.to(device)
            self.models.append(model1)
    
            model2 = pretrained.get_model('htdemucs')
            model2.to(device)
            self.models.append(model2)
    
            model3 = pretrained.get_model('htdemucs_6s')
            model3.to(device)
            self.models.append(model3)
    
            model4 = pretrained.get_model('hdemucs_mmi')
            model4.to(device)
            self.models.append(model4)
    
            if 0:
                for model in self.models:
                  pass
                  # print(model.sources)
            '''
            ['drums', 'bass', 'other', 'vocals']
            ['drums', 'bass', 'other', 'vocals']
            ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano']
            ['drums', 'bass', 'other', 'vocals']
            '''

        if device == 'cpu':
            chunk_size = 200000000
            providers = ["CPUExecutionProvider"]
        else:
            chunk_size = 1000000
            providers = ["CUDAExecutionProvider"]
        if 'chunk_size' in options:
            chunk_size = int(options['chunk_size'])

        # MDX-B model 1 initialization
        self.chunk_size = chunk_size
        self.mdx_models1 = get_models('tdf_extra', load=False, device=device, vocals_model_type=2)
        model_path_onnx1 = model_folder + 'UVR-MDX-NET-Voc_FT.onnx'
        remote_url_onnx1 = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx'
        if not os.path.isfile(model_path_onnx1):
            torch.hub.download_url_to_file(remote_url_onnx1, model_path_onnx1)
        # print('Model path: {}'.format(model_path_onnx1))
        # print('Device: {} Chunk size: {}'.format(device, chunk_size))
        self.infer_session1 = ort.InferenceSession(
            model_path_onnx1,
            providers=providers,
            provider_options=[{"device_id": 0}],
        )
        
        if self.single_onnx is False:
            # MDX-B model 2  initialization
            self.chunk_size = chunk_size
            self.mdx_models2 = get_models('tdf_extra', load=False, device=device, vocals_model_type=3)
            root_path = os.path.dirname(os.path.realpath(__file__)) + '/'
            model_path_onnx2 = model_folder + 'UVR_MDX_Instr_HQ3.onnx'
            remote_url_onnx2 = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx'
            if not os.path.isfile(model_path_onnx2):
                torch.hub.download_url_to_file(remote_url_onnx2, model_path_onnx2)
            # print('Model path: {}'.format(model_path_onnx2))
            # print('Device: {} Chunk size: {}'.format(device, chunk_size))
            self.infer_session2 = ort.InferenceSession(
                model_path_onnx2,
                providers=providers,
                provider_options=[{"device_id": 0}],
            )
        

        self.device = device
        pass
        
    @property
    def instruments(self):

        if options['vocals_only'] is False:
            return ['bass', 'drums', 'other', 'vocals']
        else:
            return ['vocals']

    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)
    def print_memory_usage(self,message):
        return 
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)    # 指定GPU的id
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        now = datetime.datetime.now()  # 获取当前时间
        print(f"当前时间: {now.strftime('%H:%M:%S')}")
        print(f"{message}-->当前分配显存: {int(meminfo.used / 1024**2)} MiB\n")
        
    def separate_music_file_voice(
            self,
            mixed_sound_array,
            sample_rate,
            current_file_number=0,
            total_files=0,
    ):
        """
        Implements the sound separation for a single sound file
        Inputs: Outputs from soundfile.read('mixture.wav')
            mixed_sound_array
            sample_rate

        Outputs:
            separated_music_arrays: Dictionary numpy array of each separated instrument
            output_sample_rates: Dictionary of sample rates separated sequence
        """

        # print('Update percent func: {}'.format(update_percent_func))
        self.print_memory_usage("Step1")
        separated_music_arrays = {}
        output_sample_rates = {}

        audio = np.expand_dims(mixed_sound_array.T, axis=0)
        audio = torch.from_numpy(audio).type('torch.FloatTensor').to(self.device)
        self.print_memory_usage("Step2")
        overlap_demucs = self.overlap_demucs
        overlap_MDX = self.overlap_MDX
        shifts = 0
        overlap = overlap_demucs

        self.print_memory_usage("Step3")
        sources3 = demix_full_mdx23c(mixed_sound_array.T, self.device)
        self.print_memory_usage("Step4")
        vocals3 = (match_array_shapes(sources3['Vocals'], mixed_sound_array.T) \
                + (mixed_sound_array.T - match_array_shapes(sources3['Instrumental'], mixed_sound_array.T))) / 2
        self.print_memory_usage("Step5")
        ####################################################################################################################################demix_full突然占用显存
        del sources3['Vocals'], sources3['Instrumental']
        torch.cuda.empty_cache()
        gc.collect()

        # print('Processing vocals with UVR-MDX-VOC-FT...')
        overlap = overlap_MDX
        sources1 = 0.5 * demix_full(
            mixed_sound_array.T,
            self.device,
            self.chunk_size,
            self.mdx_models1,
            1,
            overlap=overlap,
            bigshifts=options['bigshifts']
        )[0]
        self.print_memory_usage("Step6")
        sources1 += 0.5 * -demix_full(
            -mixed_sound_array.T,
            self.device,
            self.chunk_size,
            self.mdx_models1,
            1,
            overlap=overlap,
            bigshifts=options['bigshifts']
        )[0]
        vocals_mdxb1 = sources1 * 1.021
        self.print_memory_usage("Step7")
        ####################################################################################################################################
        # del self.infer_session1
        # del self.mdx_models1
        # torch.cuda.empty_cache()
        # gc.collect()

        sources2 = 0.5 * -demix_full(
            -mixed_sound_array.T,
            self.device,
            self.chunk_size,
            self.mdx_models2,
            2,
            overlap=overlap,
            bigshifts=options['bigshifts']//2
        )[0]
        self.print_memory_usage("Step8")
        sources2 += 0.5 * demix_full(
            mixed_sound_array.T,
            self.device,
            self.chunk_size,
            self.mdx_models2,
            2,
            overlap=overlap,
            bigshifts=options['bigshifts']//2
        )[0]
        
        # it's instrumental so need to invert
        instrum_mdxb2 = sources2
        vocals_mdxb2 = mixed_sound_array.T - (instrum_mdxb2 * 1.022)
        self.print_memory_usage("Step9") ####################################################################################################################################
        del self.infer_session2
        del self.mdx_models2
        torch.cuda.empty_cache()
        # print('Processing vocals: DONE!')
        gc.collect()
        # Ensemble vocals for MDX and Demucs
        weights = np.array([options["weight_VOCFT"], options["weight_MDXv3"], options["weight_HQ3"]])
        vocals = (lr_filter((weights[0] * vocals_mdxb1.T + weights[1] * vocals3.T + weights[2] * vocals_mdxb2.T) / weights.sum(), 14000, 'lowpass') \
                + lr_filter(vocals3.T, 14000, 'highpass')) * 1.0074 # to confirm
        self.print_memory_usage("Step10")

        # vocals
        separated_music_arrays['vocals'] = vocals
        output_sample_rates['vocals'] = sample_rate
        
        self.print_memory_usage("Step11")
        return separated_music_arrays, output_sample_rates    
    
    
    def separate_music_file(
            self,
            mixed_sound_array,
            sample_rate,
            current_file_number=0,
            total_files=0,
    ):
        """
        Implements the sound separation for a single sound file
        Inputs: Outputs from soundfile.read('mixture.wav')
            mixed_sound_array
            sample_rate

        Outputs:
            separated_music_arrays: Dictionary numpy array of each separated instrument
            output_sample_rates: Dictionary of sample rates separated sequence
        """

        # print('Update percent func: {}'.format(update_percent_func))
        
        separated_music_arrays = {}
        output_sample_rates = {}

        audio = np.expand_dims(mixed_sound_array.T, axis=0)
        audio = torch.from_numpy(audio).type('torch.FloatTensor').to(self.device)

        overlap_demucs = self.overlap_demucs
        overlap_MDX = self.overlap_MDX
        shifts = 0
        overlap = overlap_demucs
        """
        # Get Demics vocal only
        print('Processing vocals with Demucs_ft...')
        model = self.model_vocals_only
        shifts = 0
        overlap = overlap_demucs
        vocals_demucs = 0.5 * apply_model(model, audio, shifts=shifts, overlap=overlap)[0][3].cpu().numpy() \
                  + 0.5 * -apply_model(model, -audio, shifts=shifts, overlap=overlap)[0][3].cpu().numpy()
        
        model_vocals = model.cpu()
        del model_vocals
        """
        # print('Processing vocals with MDXv3 demo model...')
        sources3 = demix_full_mdx23c(mixed_sound_array.T, self.device)
        
        vocals3 = (match_array_shapes(sources3['Vocals'], mixed_sound_array.T) \
                + (mixed_sound_array.T - match_array_shapes(sources3['Instrumental'], mixed_sound_array.T))) / 2
        
        #sf.write("vocals3.wav", sources3['Vocals'].T, 44100)
        #sf.write("instru3.wav", sources3['Instrumental'].T, 44100)
        
        # sf.write("vocals3.wav", vocals3.T, 44100)
        del sources3['Vocals'], sources3['Instrumental']
        torch.cuda.empty_cache()
        gc.collect()

        # print('Processing vocals with UVR-MDX-VOC-FT...')
        overlap = overlap_MDX
        sources1 = 0.5 * demix_full(
            mixed_sound_array.T,
            self.device,
            self.chunk_size,
            self.mdx_models1,
            self.infer_session1,
            overlap=overlap,
            bigshifts=options['bigshifts']
        )[0]
        sources1 += 0.5 * -demix_full(
            -mixed_sound_array.T,
            self.device,
            self.chunk_size,
            self.mdx_models1,
            self.infer_session1,
            overlap=overlap,
            bigshifts=options['bigshifts']
        )[0]
        vocals_mdxb1 = sources1 * 1.021
        # sf.write("vocals_mdxb1.wav", vocals_mdxb1.T, 44100)

        del self.infer_session1
        del self.mdx_models1
        torch.cuda.empty_cache()
        gc.collect()
        # print('Processing vocals with UVR-MDX-HQ3-Instr...')

        sources2 = 0.5 * -demix_full(
            -mixed_sound_array.T,
            self.device,
            self.chunk_size,
            self.mdx_models2,
            self.infer_session2,
            overlap=overlap,
            bigshifts=options['bigshifts']//2
        )[0]
        sources2 += 0.5 * demix_full(
            mixed_sound_array.T,
            self.device,
            self.chunk_size,
            self.mdx_models2,
            self.infer_session2,
            overlap=overlap,
            bigshifts=options['bigshifts']//2
        )[0]

        # it's instrumental so need to invert
        instrum_mdxb2 = sources2
        vocals_mdxb2 = mixed_sound_array.T - (instrum_mdxb2 * 1.022)
        
        #  sf.write("vocals_mdxb2.wav", vocals_mdxb2.T, 44100)
        
        del self.infer_session2
        del self.mdx_models2
        torch.cuda.empty_cache()
        # print('Processing vocals: DONE!')
        gc.collect()
        # Ensemble vocals for MDX and Demucs
        weights = np.array([options["weight_VOCFT"], options["weight_MDXv3"], options["weight_HQ3"]])
        vocals = (lr_filter((weights[0] * vocals_mdxb1.T + weights[1] * vocals3.T + weights[2] * vocals_mdxb2.T) / weights.sum(), 14000, 'lowpass') \
                + lr_filter(vocals3.T, 14000, 'highpass')) * 1.0074 # to confirm

        # Generate instrumental
        instrum = mixed_sound_array - vocals
        
        if options['vocals_only'] is False:
            # print('Starting Demucs processing...')
            audio = np.expand_dims(instrum.T, axis=0)
            audio = torch.from_numpy(audio).type('torch.FloatTensor').to(self.device)
    
            all_outs = []
            # print('Processing with htdemucs_ft...')
            i = 0
            overlap = overlap_demucs
            model = pretrained.get_model('htdemucs_ft')
            model.to(self.device)
            out = 0.5 * apply_model(model, audio, shifts=shifts, overlap=overlap)[0].cpu().numpy() \
                  + 0.5 * -apply_model(model, -audio, shifts=shifts, overlap=overlap)[0].cpu().numpy()
       
            out[0] = self.weights_drums[i] * out[0]
            out[1] = self.weights_bass[i] * out[1]
            out[2] = self.weights_other[i] * out[2]
            out[3] = self.weights_vocals[i] * out[3]
            all_outs.append(out)
            model = model.cpu()
            del model
            gc.collect()
            i = 1
            # print('Processing with htdemucs...')
            overlap = overlap_demucs
            model = pretrained.get_model('htdemucs')
            model.to(self.device)
            out = 0.5 * apply_model(model, audio, shifts=shifts, overlap=overlap)[0].cpu().numpy() \
                  + 0.5 * -apply_model(model, -audio, shifts=shifts, overlap=overlap)[0].cpu().numpy()
    
            out[0] = self.weights_drums[i] * out[0]
            out[1] = self.weights_bass[i] * out[1]
            out[2] = self.weights_other[i] * out[2]
            out[3] = self.weights_vocals[i] * out[3]
            all_outs.append(out)
            model = model.cpu()
            del model
            gc.collect()
            i = 2
            # print('Processing with htdemucs_6s...')
            overlap = overlap_demucs
            model = pretrained.get_model('htdemucs_6s')
            model.to(self.device)
            out = apply_model(model, audio, shifts=shifts, overlap=overlap)[0].cpu().numpy()
       
            # More stems need to add
            out[2] = out[2] + out[4] + out[5]
            out = out[:4]
            out[0] = self.weights_drums[i] * out[0]
            out[1] = self.weights_bass[i] * out[1]
            out[2] = self.weights_other[i] * out[2]
            out[3] = self.weights_vocals[i] * out[3]
            all_outs.append(out)
            model = model.cpu()
            del model
            gc.collect()
            i = 3
            # print('Processing with htdemucs_mmi...')
            model = pretrained.get_model('hdemucs_mmi')
            model.to(self.device)
            out = 0.5 * apply_model(model, audio, shifts=shifts, overlap=overlap)[0].cpu().numpy() \
                  + 0.5 * -apply_model(model, -audio, shifts=shifts, overlap=overlap)[0].cpu().numpy()
       
            out[0] = self.weights_drums[i] * out[0]
            out[1] = self.weights_bass[i] * out[1]
            out[2] = self.weights_other[i] * out[2]
            out[3] = self.weights_vocals[i] * out[3]
            all_outs.append(out)
            model = model.cpu()
            del model
            gc.collect()
            out = np.array(all_outs).sum(axis=0)
            out[0] = out[0] / self.weights_drums.sum()
            out[1] = out[1] / self.weights_bass.sum()
            out[2] = out[2] / self.weights_other.sum()
            out[3] = out[3] / self.weights_vocals.sum()

            # other
            res = mixed_sound_array - vocals - out[0].T - out[1].T
            res = np.clip(res, -1, 1)
            separated_music_arrays['other'] = (2 * res + out[2].T) / 3.0
            output_sample_rates['other'] = sample_rate
    
            # drums
            res = mixed_sound_array - vocals - out[1].T - out[2].T
            res = np.clip(res, -1, 1)
            separated_music_arrays['drums'] = (res + 2 * out[0].T.copy()) / 3.0
            output_sample_rates['drums'] = sample_rate
    
            # bass
            res = mixed_sound_array - vocals - out[0].T - out[2].T
            res = np.clip(res, -1, 1)
            separated_music_arrays['bass'] = (res + 2 * out[1].T) / 3.0
            output_sample_rates['bass'] = sample_rate
    
            bass = separated_music_arrays['bass']
            drums = separated_music_arrays['drums']
            other = separated_music_arrays['other']
    
            separated_music_arrays['other'] = mixed_sound_array - vocals - bass - drums
            separated_music_arrays['drums'] = mixed_sound_array - vocals - bass - other
            separated_music_arrays['bass'] = mixed_sound_array - vocals - drums - other
            
            
        # vocals
        separated_music_arrays['vocals'] = vocals
        output_sample_rates['vocals'] = sample_rate
        
        # instrum
        separated_music_arrays['instrum'] = instrum

        return separated_music_arrays, output_sample_rates


def predict_with_model(options):

    output_format = options['output_format']

    for input_audio in options['input_audio']:
        if not os.path.isfile(input_audio):
            print('Error. No such file: {}. Please check path!'.format(input_audio))
            return
    output_folder = options['output_folder']
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    model = None
    model = EnsembleDemucsMDXMusicSeparationModel(options)

    for i, input_audio in enumerate(options['input_audio']):
        # print('Go for: {}'.format(input_audio))
        audio, sr = librosa.load(input_audio, mono=False, sr=44100)
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=0)
        # print("Input audio: {} Sample rate: {}".format(audio.shape, sr))
        result, sample_rates = model.separate_music_file_voice(audio.T, sr, i, len(options['input_audio']))
        for instrum in model.instruments:
            output_name = os.path.splitext(os.path.basename(input_audio))[0] + '_{}.wav'.format(instrum)
            sf.write(output_folder + '/' + output_name, result[instrum], sample_rates[instrum], subtype=output_format)

#         inst = result['instrum']
#         output_name = os.path.splitext(os.path.basename(input_audio))[0] + '_{}.wav'.format('instrum')
#         sf.write(output_folder + '/' + output_name, inst, sr, subtype=output_format)
#         # print('File created: {}'.format(output_folder + '/' + output_name))
        
#         if options['vocals_only'] is False:
#             # instrumental part 2
#             inst2 = (result['bass'] + result['drums'] + result['other']) # 1.004
#             output_name = os.path.splitext(os.path.basename(input_audio))[0] + '_{}.wav'.format('instrum2')
#             sf.write(output_folder + '/' + output_name, inst2, sr, subtype=output_format)
#             # print('File created: {}'.format(output_folder + '/' + output_name))


# Linkwitz-Riley filter
def lr_filter(audio, cutoff, filter_type, order=6, sr=44100):
    audio = audio.T
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order//2, normal_cutoff, btype=filter_type, analog=False)
    filtered_audio = signal.filtfilt(b, a, audio)
    return filtered_audio.T

# SRS
def change_sr(data, up, down):
    data = data.T
    # print(f"SRS input audio shape: {data.shape}")
    new_data = resample_poly(data, up, down)
    # print(f"SRS output audio shape: {new_data.shape}")
    return new_data.T

# Lowpass filter
def lp_filter(cutoff, data, sample_rate):
    b = signal.firwin(1001, cutoff, fs=sample_rate)
    filtered_data = signal.filtfilt(b, [1.0], data)
    return filtered_data

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def match_array_shapes(array_1:np.ndarray, array_2:np.ndarray):
    if array_1.shape[1] > array_2.shape[1]:
        array_1 = array_1[:,:array_2.shape[1]] 
    elif array_1.shape[1] < array_2.shape[1]:
        padding = array_2.shape[1] - array_1.shape[1]
        array_1 = np.pad(array_1, ((0,0), (0,padding)), 'constant', constant_values=0)
    return array_1
import argparse
from time import time

def get_ort():
    from concurrent.futures import ThreadPoolExecutor
    from multiprocessing import Process, Lock
    import os
    from tqdm import tqdm
    lock = Lock()  # 创建一个进程间共享的锁
    input_audio = '/root/autodl-tmp/'  
    # input_audio = '/root/autodl-tmp/DATA/INPUT/origin_1.wav'  
    output_folder = '/root/autodl-tmp/'
    options = {
        "input_audio": [],
        "output_folder": "",
        "use_cpu": False,
        "overlap_demucs": 0.6,
        "overlap_MDX": 0,
        "overlap_MDXv3": 20,
        "weight_MDXv3": 8,
        "weight_VOCFT": 3,
        "weight_HQ3": 2,
        "single_onnx": False,
        "chunk_size": 1000000,
        "large_gpu": True,
        "bigshifts": 6,
        "vocals_only": True,
        "output_format": 'PCM_16',
        "lock":lock,
    }
    options["input_audio"] = [input_audio]
    options["output_folder"] = output_folder
    options = copy_options(**options)
    output_format = options['output_format']

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    model = None
    model = EnsembleDemucsMDXMusicSeparationModel(options)
    ort1 = model.infer_session1
    ort2 = model.infer_session2
    del model
    torch.cuda.empty_cache()
    gc.collect()
    init_data = [np.random.rand(i, 4, 3072, 256).astype(np.float32) for i in range(4,0,-1)]
    for i in range(8):
        temp = init_data[int(i/2)]
        ort1.run(None, {'input':temp})[0];
        ort2.run(None, {'input':temp})[0];
    return ort1,ort2

def copy_options(input_audio, output_folder, use_cpu=False, overlap_demucs=0.6, overlap_MDX=0.5, overlap_MDXv3=10, weight_MDXv3=6, weight_VOCFT=5, weight_HQ3=2, single_onnx=False, chunk_size=1000000, large_gpu=False, bigshifts=1, vocals_only=False, output_format="FLOAT", lock=None):
    start_time = time()
    # print("started!")
    # print(f"正在处理{os.path.basename(input_audio)}")
    global options
    
    options = {
        "input_audio": input_audio,
        "output_folder": output_folder,
        "cpu": use_cpu,
        "overlap_demucs": overlap_demucs,
        "overlap_MDX": overlap_MDX,
        "overlap_MDXv3": overlap_MDXv3,
        "weight_MDXv3": weight_MDXv3,
        "weight_VOCFT": weight_VOCFT,
        "weight_HQ3": weight_HQ3,
        "single_onnx": single_onnx,
        "chunk_size": chunk_size,
        "large_gpu": large_gpu,
        "bigshifts": bigshifts,
        "vocals_only": vocals_only,
        "output_format": output_format,
        "lock":lock,
    }
    return options

def run_music_separator(input_audio, output_folder, use_cpu=False, overlap_demucs=0.6, overlap_MDX=0.5, overlap_MDXv3=10, weight_MDXv3=6, weight_VOCFT=5, weight_HQ3=2, single_onnx=False, chunk_size=1000000, large_gpu=False, bigshifts=1, vocals_only=False, output_format="FLOAT", lock=None,output_queue=None, inference_input_queue=None,worker_id=None):
    start_time = time()
    # print("started!")
    # print(f"正在处理{os.path.basename(input_audio)}")
    global options
    
    options = {
        "input_audio": input_audio,
        "output_folder": output_folder,
        "cpu": use_cpu,
        "overlap_demucs": overlap_demucs,
        "overlap_MDX": overlap_MDX,
        "overlap_MDXv3": overlap_MDXv3,
        "weight_MDXv3": weight_MDXv3,
        "weight_VOCFT": weight_VOCFT,
        "weight_HQ3": weight_HQ3,
        "single_onnx": single_onnx,
        "chunk_size": chunk_size,
        "large_gpu": large_gpu,
        "bigshifts": bigshifts,
        "vocals_only": vocals_only,
        "output_format": output_format,
        "lock":lock,
        "output_queue":output_queue, 
        "inference_input_queue":inference_input_queue,
        "worker_id":worker_id,
    }
    predict_with_model(options)


