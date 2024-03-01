from flask import Flask, request, send_from_directory, Response, jsonify
import boto3
import uuid
import asyncio
from scipy.io import wavfile

from tortoise.utils.text import split_and_recombine_text


import os

import sys

##THIS IS TTS WITH ENHANCEN AUDIO and transcription - NEED resemble_enhance MODULE TO USE IT



import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from resemble_enhance.enhancer.inference import denoise, enhance

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#from faster_whisper import WhisperModel



import random
random.seed(0)

import numpy as np
np.random.seed(0)

# load packages
import time
import yaml
from munch import Munch
import librosa
import nltk

from nltk.tokenize import word_tokenize

from models import *
from utils import *
from text_utils import TextCleaner

import soundfile as sf
import json
import base64
import GPUtil

app = Flask(__name__)


# Used to return download link to user
WEB_URL = 'https://pickles.io'


nltk.download('punkt')

textclenaer = TextCleaner()


to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(path):
    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load phonemizer
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)

config = yaml.safe_load(open("Models/LibriTTS/config.yml"))

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

# load BERT model
from Utils.PLBERT.util import load_plbert
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

model_params = recursive_munch(config['model_params'])
model = build_model(model_params, text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

params_whole = torch.load("Models/LibriTTS/epochs_2nd_00020.pth", map_location='cpu')
params = params_whole['net']

for key in model:
    if key in params:
        print('%s loaded' % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            model[key].load_state_dict(new_state_dict, strict=False)
#             except:
#                 _load(params[key], model[key])
_ = [model[key].eval() for key in model]

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
    clamp=False
)


def inference(text, ref_s, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device),
                                          embedding=bert_dur,
                                          embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                             num_steps=diffusion_steps).squeeze(1)


        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en,
                                         s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)


        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr,
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))


    return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later

def LFinference(text, s_prev, ref_s, alpha = 0.3, beta = 0.7, t = 0.7, diffusion_steps=5, embedding_scale=1):
  text = text.strip()
  ps = global_phonemizer.phonemize([text])
  ps = word_tokenize(ps[0])
  ps = ' '.join(ps)
  ps = ps.replace('``', '"')
  ps = ps.replace("''", '"')

  tokens = textclenaer(ps)
  tokens.insert(0, 0)
  tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

  with torch.no_grad():
      input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
      text_mask = length_to_mask(input_lengths).to(device)

      t_en = model.text_encoder(tokens, input_lengths, text_mask)
      bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
      d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

      s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device),
                                        embedding=bert_dur,
                                        embedding_scale=embedding_scale,
                                          features=ref_s, # reference from the same speaker as the embedding
                                            num_steps=diffusion_steps).squeeze(1)

      if s_prev is not None:
          # convex combination of previous and current style
          s_pred = t * s_prev + (1 - t) * s_pred

      s = s_pred[:, 128:]
      ref = s_pred[:, :128]

      ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
      s = beta * s + (1 - beta)  * ref_s[:, 128:]

      s_pred = torch.cat([ref, s], dim=-1)

      d = model.predictor.text_encoder(d_en,
                                        s, input_lengths, text_mask)

      x, _ = model.predictor.lstm(d)
      duration = model.predictor.duration_proj(x)

      duration = torch.sigmoid(duration).sum(axis=-1)
      pred_dur = torch.round(duration.squeeze()).clamp(min=1)


      pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
      c_frame = 0
      for i in range(pred_aln_trg.size(0)):
          pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
          c_frame += int(pred_dur[i].data)

      # encode prosody
      en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
      if model_params.decoder.type == "hifigan":
          asr_new = torch.zeros_like(en)
          asr_new[:, :, 0] = en[:, :, 0]
          asr_new[:, :, 1:] = en[:, :, 0:-1]
          en = asr_new

      F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

      asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
      if model_params.decoder.type == "hifigan":
          asr_new = torch.zeros_like(asr)
          asr_new[:, :, 0] = asr[:, :, 0]
          asr_new[:, :, 1:] = asr[:, :, 0:-1]
          asr = asr_new

      out = model.decoder(asr,
                              F0_pred, N_pred, ref.squeeze().unsqueeze(0))


  return out.squeeze().cpu().numpy()[..., :-100], s_pred # weird pulse at the end of the model, need to be fixed later

def STinference(text, ref_s, ref_text, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    ref_text = ref_text.strip()
    ps = global_phonemizer.phonemize([ref_text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    ref_tokens = textclenaer(ps)
    ref_tokens.insert(0, 0)
    ref_tokens = torch.LongTensor(ref_tokens).to(device).unsqueeze(0)


    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        ref_input_lengths = torch.LongTensor([ref_tokens.shape[-1]]).to(device)
        ref_text_mask = length_to_mask(ref_input_lengths).to(device)
        ref_bert_dur = model.bert(ref_tokens, attention_mask=(~ref_text_mask).int())
        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device),
                                          embedding=bert_dur,
                                          embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                             num_steps=diffusion_steps).squeeze(1)


        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en,
                                         s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)


        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr,
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))


    return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later



noise = torch.randn(1,1,256).to(device)
# for k, path in reference_dicts.items():
ref_her = compute_style("SamVoice.wav")
path = "SamVoice.wav"



def process_audio(path, output_path, solver="Midpoint", nfe=64, tau=0.5, denoising=False):
    if not path:
        return "Invalid input path."

    solver = solver.lower()
    nfe = int(nfe)
    lambd = 0.9 if denoising else 0.1

    # Load and process the audio
    dwav, sr = torchaudio.load(path)
    dwav = dwav.mean(dim=0)

    wav1, new_sr = denoise(dwav, sr, "cuda")
    # wav1, new_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau)

    wav1 = wav1.cpu().unsqueeze(0)

    # Save the processed audio to a file
    torchaudio.save(output_path, wav1, new_sr)
    return new_sr


def text_to_speech_with_options(text, enhance=False, gpu_code=""):
    # Check if text is longer than 300 characters
    if len(text) > 300:
        # Split text into sentences and process each individually
        sentences = split_and_recombine_text(text)
        wavs = []
        s_prev = None
        for text1 in sentences:
            if text1.strip() == "":
                continue
            text1 += '.'  # add the period back

            wav, s_prev = LFinference(text1,
                                      s_prev,
                                      ref_her,
                                      alpha=0.2,
                                      beta=0.9,
                                      t=0.7,
                                      diffusion_steps=8, 
                                      embedding_scale=1.3)
            wavs.append(wav)
        
        # Concatenate all audio segments
        wav = np.concatenate(wavs)
    else:
        # Original processing for shorter texts
        wav = inference(text, ref_her, alpha=0.2, beta=0.7, diffusion_steps=5, embedding_scale=1.1)
        print('wav.shape prior to enhance:', wav.shape)

    
    file_key = str(uuid.uuid4()) + '.wav'
    output_filename = 'files_for_download/' + file_key
    sf.write(output_filename, wav, 24000)
    final_audio_filename = output_filename


    # Enhance Audio if requested
    if enhance:
        enhanced_output_filename = 'files_for_download/enhanced_' + file_key
        _new_sr = process_audio(output_filename, output_path=enhanced_output_filename, denoising=True)
        final_audio_filename = enhanced_output_filename

    print('wav.shape after enhance:', wav.shape)
        
    # Construct file url
    file_url = WEB_URL + '/download/' + gpu_code + '-' + final_audio_filename
    file_url = file_url.replace('files_for_download/', '')    # drop this element from the URL
    print('file_url:', file_url)
    

    # Return the URL and transcription data
    response = {"file_url": file_url}
    return response





def text_to_speech_with_options_return_file(text, enhance=False, gpu_code=""):
    # Check if text is longer than 300 characters
    if len(text) > 300:
        # Split text into sentences and process each individually
        sentences = split_and_recombine_text(text)
        wavs = []
        s_prev = None
        for text1 in sentences:
            if text1.strip() == "":
                continue
            text1 += '.'  # add the period back

            wav, s_prev = LFinference(text1,
                                      s_prev,
                                      ref_her,
                                      alpha=0.2,
                                      beta=0.9,
                                      t=0.7,
                                      diffusion_steps=8, 
                                      embedding_scale=1.3)
            wavs.append(wav)
        
        # Concatenate all audio segments
        wav = np.concatenate(wavs)
    else:
        # Original processing for shorter texts
        wav = inference(text, ref_her, alpha=0.2, beta=0.7, diffusion_steps=5, embedding_scale=1.1)
        print('wav.shape prior to enhance:', wav.shape)

    
    file_key = str(uuid.uuid4()) + '.wav'
    output_filename = 'files_for_download/' + file_key
    sf.write(output_filename, wav, 24000)
    final_audio_filename = output_filename


    # Enhance Audio if requested
    if enhance:
        enhanced_output_filename = 'files_for_download/enhanced_' + file_key
        _new_sr = process_audio(output_filename, output_path=enhanced_output_filename, denoising=True)
        final_audio_filename = enhanced_output_filename

    sample_rate, wav = wavfile.read(final_audio_filename)
    

    # Return the file as bytes
    wav_bytes = wav.tobytes()
    response = {
        "file_bytes": wav.tobytes(),
        "sample_rate": sample_rate
    }
    
    return response



def stream_generator_text_to_speech_with_options(
    text, 
    enhance=False,
    desired_stream_section_length=100,
    max_stream_section_length=150
):

    sample_rate = 24_000
    count_bytes_per_chunk = 12_000

    # We set count_bytes_per_chunk so each chunk is 1/4 seconds
    # Multiply by two as each sample is 2 bytes (16 bit integers)
    # count_bytes_per_chunk = int(samplerate * 2 / 4)
    ## Though if enhanced=True there will be more chunks per second as samplerate becomes 44100
    
    # Split into smaller sections than non-streaming
    if len(text) > max_stream_section_length:
        
        # Split text into sentences and process each individually
        sentences = split_and_recombine_text(
            text, 
            desired_length=desired_stream_section_length, 
            max_length=max_stream_section_length
        )
        
        sentences = [s + '.' for s in sentences]   # add the period back
    else:
        sentences = [text]
        
    s_prev = None
    bytes_to_stream = None
    for sentence in sentences:
        if sentence.strip() == "":
            continue

        t1 = time.time()
        wav, s_prev = LFinference(sentence,
                                  s_prev,
                                  ref_her,
                                  alpha=0.2,
                                  beta=0.9,
                                  t=0.7,
                                  diffusion_steps=8, 
                                  embedding_scale=1.3)
        t2 = time.time()
        print(t2 - t1, 'seconds to run inference for char length', len(sentence))
        print('wav.shape prior to enhance:', wav.shape)
        print(type(wav), 'is wav type')

        # Enhance Audio if requested
        if enhance:

            t1 = time.time()
            
            # saving file so can stream out
            file_key = str(uuid.uuid4()) + '.wav'
            output_filename = 'files_for_download/' + file_key
            sf.write(output_filename, wav, 24000)
            final_audio_filename = output_filename
            
            enhanced_output_filename = 'files_for_download/enhanced_' + file_key
            new_sr = process_audio(output_filename, output_path=enhanced_output_filename, denoising=True)
            _sr, wav = wavfile.read(enhanced_output_filename)
            sample_rate = new_sr

            t2 = time.time()
            print(t2 - t1, 'seconds to enhance for char length', len(sentence))
            print('sampling rate _sr:', _sr, 'new_sr', new_sr, 'sample_rate', sample_rate)

        
        # If not enhanced write and read again to convert to same sample rate
        else:
            file_key = str(uuid.uuid4()) + '.wav'
            output_filename = 'files_for_download/' + file_key
            sf.write(output_filename, wav, 24000)
            _sr, wav = wavfile.read(output_filename)
            print('_sr not enhanced:', _sr)

        ## Append to bytes array for streaming
        print('wav.shape prior to converting to bytes:', wav.shape)
        wav_bytes = wav.tobytes()

        if bytes_to_stream is not None:
            bytes_to_stream = bytes_to_stream + wav_bytes
        else:
            bytes_to_stream = wav_bytes

        print(len(bytes_to_stream), 'bytes_to_stream')
        print(count_bytes_per_chunk, 'count_bytes_per_chunk')

        # yield as bytes: loop until too few bytes for final chunk
        for i in range(0, len(bytes_to_stream), count_bytes_per_chunk):
            
            payload_for_yield = bytes_to_stream[:count_bytes_per_chunk]
            
            if len(payload_for_yield) == count_bytes_per_chunk:

                # drop the bytes being yielded from front of the list
                bytes_to_stream = bytes_to_stream[count_bytes_per_chunk:]
                
                yield payload_for_yield
    
    # when all sentences are made and yielded: send the remaining chunk of bytes
    yield payload_for_yield

    


@app.route('/api/stream', methods=['POST'])  
def stream_audio():

    job = request.json
    job_input = job["input"]
    print('job_input:', job_input)

    # Extract parameters from job input
    text = job_input.get("text", "")
    enhance = job_input.get("enhance", True)
    desired_stream_section_length = job_input.get("desired_stream_section_length", 100)
    max_stream_section_length = job_input.get("max_stream_section_length", 150)
    
    return Response(
        stream_generator_text_to_speech_with_options(
            text, 
            enhance, 
            desired_stream_section_length, 
            max_stream_section_length
        ), mimetype='audio/wav')  


@app.route('/')
def hello_world():
    return 'Hello, World from Flask!'


@app.route('/get_gpu_load')
def get_gpu_load():
    gpu = GPUtil.getGPUs()[0]
    return {"gpu_load": gpu.load}



@app.route('/api', methods=['POST'])
def handle_request():
    try:
        job = request.json
        if not job:
            return jsonify({"error": "No JSON data provided"}), 400

        job_input = job.get("input")
        if not job_input:
            return jsonify({"error": "Missing 'input' key in JSON data"}), 400

        text = job_input.get("text", "")
        enhance = job_input.get("enhance", False)
        gpu_code = job_input.get("gpu_code", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        response = text_to_speech_with_options(text, enhance, gpu_code)
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/api_return_file', methods=['POST'])
def handle_request():
    try:
        job = request.json
        if not job:
            return jsonify({"error": "No JSON data provided"}), 400

        job_input = job.get("input")
        if not job_input:
            return jsonify({"error": "Missing 'input' key in JSON data"}), 400

        text = job_input.get("text", "")
        enhance = job_input.get("enhance", False)
        gpu_code = job_input.get("gpu_code", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        response = text_to_speech_with_options_return_file(text, enhance, gpu_code)
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route('/download/<file_name>', methods=['GET'])
def download(file_name):
    try:
        return send_from_directory('files_for_download', file_name, as_attachment=True)
    except FileNotFoundError:
        return {"error", "File not found"} 


if __name__ == '__main__':
    app.run(debug=False)
