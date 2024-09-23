from src.metrics.metrics import compute_decay, Metrics
from src.utils import read_audio_file, write_audio_file
import src.utils as utils
import argparse
import os, json, glob


import torch.nn as nn

import numpy as np
import torch
import pandas as pd
import torchaudio

import auraloss
#from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
#from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality

def save_audio_file_torch(file_path, wavform, sample_rate = 48000, rescale = True):
    if rescale:
        wavform = wavform/torch.max(wavform)*0.9
    torchaudio.save(file_path, wavform, sample_rate)


def angle_between_vectors(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.rad2deg(np.arccos(dot_product))
    if angle < 90:
        return angle
    else:
        return 180 - angle

def load_testcase(sample_dir: str, args):
    # [1] Load metadata
    metadata_path = os.path.join(sample_dir, 'metadata.json')
    with open(metadata_path, 'rb') as f:
        metadata = json.load(f)

    # [2] Load mixture
    mixture_path = os.path.join(sample_dir, 'mixture.wav')
    mixture = read_audio_file(mixture_path, args.sr)
    #mix = []
    #for i in range(6):
    #  mix.append(read_audio_file(os.path.join(sample_dir, f'mic{i:02d}_mixed.wav'), args.sr))
    #mixture = np.array(mix)

    # [3] Load ground truth from metadata and count the number of speakers
    # in the ground truth
    gt = np.zeros((1, mixture.shape[-1])) # Single channel
    speakers = [key for key in metadata if key.startswith('voice')]
    dis_near = []
    dis_far = []
    angle_near = []
    angle_far = []

    tgt_speakers = []

    for speaker in speakers:
        speaker_distance = metadata[speaker]['dis'] / 100
        angle = metadata[speaker]['angle']  

        # If speaker is within the threshold, add s
        if speaker_distance <= args.distance_threshold:
            dis_near.append(speaker_distance)
            angle_near.append(angle)

            speaker_solo = read_audio_file(os.path.join(sample_dir, f'mic{0:02d}_{speaker}.wav'), args.sr)
            gt += speaker_solo

            tgt_speakers.append(metadata[speaker])
        else:
            dis_far.append(speaker_distance)
            angle_far.append(angle)

#    dis_diff = max(dis_near) - min(dis_far)

    return metadata, mixture, gt, tgt_speakers

def run_testcase(model, mixture: np.ndarray, device) -> np.ndarray:
    with torch.no_grad():
        # Create tensor and copy it to the device
        mixture = torch.from_numpy(mixture)
        mixture = mixture.to(device)

        # Run inference
        inputs = dict(mixture=mixture.unsqueeze(0))
        outputs = model(inputs)
        output = outputs['output'].squeeze(0)
        
        # Copy to cpu and convert to numpy array
        output = output.cpu().numpy()

        return output

def main(args: argparse.Namespace):
    device = 'cuda' if args.use_cuda else 'cpu'
    
    os.makedirs(args.output_dir, exist_ok=True)

    sample_dirs = sorted(glob.glob(os.path.join(args.test_dir, '*')))
    
    # Load model
    model = utils.load_pretrained(args.run_dir).model
    model = model.to(device)
    model.eval()

    # Initialize metrics
    snr = Metrics('snr')
    snr_i = Metrics('snr_i')
    
    si_snr = Metrics('si_snr')
    si_snr_i = Metrics('si_snr_i')
    
    si_sdr = Metrics('si_sdr')
    si_sdr_i = Metrics('si_sdr_i')

    # subjective metrics
    pesq = Metrics('PESQ')
    stoi = Metrics('STOI')


    ### new loss 
    stoi = Metrics('STOI')
    stoi = Metrics('STOI')
    

    MultiLoss = auraloss.freq.MultiResolutionSTFTLoss(w_sc = 0, w_log_mag = 0, w_lin_mag = 10, perceptual_weighting = True, output  = "full", sample_rate = 24000)
    L1loss = nn.L1Loss()
    ### PLCPALoss
    params = {"window_size": 288,
           "hop_size": 192,
           "fft_len": 288,
           "power": 0.3,
           "scale_asym": 1,
           "scale_mag": 0.9,
           "scale_phase": 0.1,
           "return_all": True
           }

    plcpa = Metrics('PLCPALoss', **params)


    #hubert = Metrics('Hubert')



    peoples = []    
    snr_ins = []
    snris = []
    sisdr_ins = []
    sisdris = []
    decays = []
    pesqs = []
    stois = []
    pesq_ins = []
    stoi_ins = []

    records = []
    num = 0

    
    specific_pos = {}

    for sample_dir in sample_dirs:
        sample_name = os.path.basename(sample_dir)

        if args.save_id >= 0:
            sample_name = "{:06d}".format(args.save_id)
            sample_dir = os.path.join(args.test_dir, sample_name)
        print(f"Sample: {sample_name}", sample_dir)
        if num > 100:
           break
        num += 1
        # Load data
        metadata, mixture, gt, tgt_speakers = load_testcase(sample_dir, args)
        n_tgt_speakers = len(tgt_speakers)

        # print(metadata)
        # Run inference
        output = run_testcase(model, mixture, device)
        peoples.append(metadata["room"])
        row = {}
        row['sample'] = sample_name
        row['room'] = metadata['room']
        row['dis']  = metadata['voice00']['dis']
        row['angle']  = 90 - metadata['voice00']['angle']
        row['tgt_speaker_ids'] = [spk['speaker_id'] for spk in tgt_speakers]
        row['tgt_speaker_distances'] = [spk['dis'] for spk in tgt_speakers]
        row['n_tgt_speakers'] = n_tgt_speakers
        

        pos_name = row['room'] + "_" + str(int(row['dis']))  + "_"+ str(int(row['angle']))
        #print('Num speakers:', n_tgt_speakers)
        if n_tgt_speakers == 0:
            # Compute decay for the case where there is no target speaker
            row['decay'] = compute_decay(est=output, mix=mixture[0:1]).item()
            print("Decay:", row['decay'])
            decays.append(row['decay'])
            if args.save_id  >= 0:
                save_audio_file_torch("./debug/mix" + sample_name + ".wav", torch.from_numpy(mixture[0:1]).float(), sample_rate = args.sr,rescale = False)
                save_audio_file_torch("./debug/est" + sample_name + ".wav", torch.from_numpy(output).float(), sample_rate = args.sr,rescale = False)
                save_audio_file_torch("./debug/gt" + sample_name + ".wav", torch.from_numpy(gt).float(), sample_rate = args.sr,rescale = False)
            
        else:
            # Compute SNR-based metrics for the case where there is at least one target speaker                       
            
            # Input SNR & SNR
            row['input_snr'] = snr(est=mixture[0:1], gt=gt, mix=mixture[0:1]).item()
            row['snri'] = snr_i(est=output, gt=gt, mix=mixture[0:1]).item()

            # Output SI-SNR & SI-SNRi
            row['input_sisnr'] = si_snr(est=mixture[0:1], gt=gt, mix=mixture[0:1]).item()
            row['sisnri'] = si_snr_i(est=output, gt=gt, mix=mixture[0:1]).item()
            
            # Input SI-SDR & SI-SDRi
            row['input_sisdr'] = si_sdr(est=mixture[0:1], gt=gt, mix=mixture[0:1]).item()
            row['sisdri'] = si_sdr_i(est=output, gt=gt, mix=mixture[0:1]).item()

            # subjective \
            row['stoi_in'] = stoi(est=mixture[0:1], gt=gt, mix=mixture[0:1]).item()
            row['pesq_in'] = pesq(est=mixture[0:1], gt=gt, mix=mixture[0:1]).item()

            row['stoi'] = stoi(est=output, gt=gt, mix=mixture[0:1]).item()
            row['pesq'] = pesq(est=output, gt=gt, mix=mixture[0:1]).item()
            

            
            gt = torch.from_numpy(gt).float()
            output = torch.from_numpy(output).float()
            mixture = torch.from_numpy(mixture).float()


            print(L1loss(output.unsqueeze(0), gt.unsqueeze(0)))
            print(MultiLoss(output.unsqueeze(0), gt.unsqueeze(0)))

            aysm_plcpa, plcpaloss, aysmloss = plcpa(est=output, gt=gt, mix=mixture[0:1])
            row["aysm_plcpa"] = aysm_plcpa.item()
            row["plcpa"] = plcpaloss.item()
            row["aysm"] = aysmloss.item()

            #gt = gt.to(device)
            #output = output.to(device)
            #mixture = mixture.to(device)
            #row["hubert"] = 0#hubert(output, gt, mixture[0:1]).item()

            #(preds, target, 16000, 'wb')
            snr_ins.append(row['input_snr'])
            snris.append(row['snri'])
            sisdr_ins.append(row['input_sisdr'])
            sisdris.append(row['sisdri'])
            pesqs.append(row['pesq'])
            stois.append(row['stoi'])
            pesq_ins.append(row['pesq_in'])
            stoi_ins.append(row['stoi_in'])

            if pos_name not in specific_pos.keys():
                temp = {}
                temp["SISDRi"] = [row['sisdri']]
                temp["PESQ"] = [row['pesq']]
                specific_pos[pos_name] = temp
            else:
                specific_pos[pos_name]["SISDRi"].append(row['sisdri'])
                specific_pos[pos_name]["PESQ"].append(row['pesq'])

            if args.save_id  >= 0:
                save_audio_file_torch("./debug/mix" + sample_name + ".wav", mixture[0:1].cpu(), sample_rate = args.sr,rescale = False)
                save_audio_file_torch("./debug/est" + sample_name + ".wav", output.cpu(), sample_rate = args.sr,rescale = False)
                save_audio_file_torch("./debug/gt" + sample_name + ".wav", gt.cpu(), sample_rate = args.sr,rescale = False)
            #print(row['hubert'], row['stoi'])
            if  row['sisdri'] < 0 or row['snri'] < 0:
                # print(sample_dir)
                print('SI-SDR:', row['input_sisdr'], row['sisdri'], "SNR: ", row['input_snr'], row['snri'])
                print("pesq_in=", row['pesq_in'], "pesq=", row['pesq'])
                print("stoi_in=", row['stoi_in'], "stoi=", row['stoi'])
                #print("plcpa=", row['aysm_plcpa'], "hubert=", row['hubert'])
        records.append(row)

        if  args.save_id  >= 0:
            break
    ### calculate the mean results


    print("DECAY = ", np.mean(decays))
    print("SNR: ", np.mean(snr_ins), np.mean(snris))
    print("SISDR: ", np.mean(sisdr_ins), np.mean(sisdris))
    print("pesq = ", np.mean(pesq_ins), np.mean(pesqs))
    print("stoi = ", np.mean(stoi_ins), np.mean(stois))
    print("stoi = ", np.mean(stoi_ins), np.mean(stois))
    
    #for name in specific_pos.keys():
    #    print(name)
    #    print("SISDRi = ", np.mean(specific_pos[name]["SISDRi"]))
    #    print("PESQ = ", np.mean(specific_pos[name]["PESQ"]))
    print(len(peoples), len(snris))
    if  args.save_id  < 0:
        ### room wise metrics
        # user_metrics = {}
        # for i in range(len(peoples)):
        #     name = peoples[i]
        #     if name not in user_metrics.keys():
        #         temp = {}
        #         temp["snri"] = []
        #         temp["sisdri"] = []
        #         temp["pesq"] = []
        #         temp["stoi"] = []
        #         user_metrics[name] = temp
        #     user_metrics[name]["snri"].append(snris[i]) 
        #     user_metrics[name]["sisdri"].append(sisdris[i])
        #     user_metrics[name]["pesq"].append(pesqs[i])
        #     user_metrics[name]["stoi"].append(stois[i])

        
        # for name in user_metrics.keys():
        #     print(name, "*"*10)
        #     print("snri = ", np.mean(user_metrics[name]["snri"]),"sisdri = ", np.mean(user_metrics[name]["sisdri"]))
        #     print("pesq = ", np.mean(user_metrics[name]["pesq"]), "stoi = ", np.mean(user_metrics[name]["stoi"]))

        # Create DataFrame from records
        results_df = pd.DataFrame.from_records(records)
        
        # Save DataFrame
        results_csv_path = os.path.join(args.output_dir, 'results.csv')
        results_df.to_csv(results_csv_path)

        # Save arguments to this script
        args_path = os.path.join(args.output_dir, 'args.json')
        with open(args_path, 'w') as f:
            json.dump(args.__dict__, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir',
                        type=str,
                        help="Path to test dataset")
    parser.add_argument('run_dir',
                        type=str,
                        help='Path to model run')
    parser.add_argument('output_dir',
                        type=str,
                        help='Path to store output files')
    
    parser.add_argument('--distance_threshold',
                        type=float,
                        default=1.0,
                        help='Distance threshold to include/exclude speakers')
    parser.add_argument('--sr',
                        type=int,
                        default=24000,
                        help='Project sampling rate')

    parser.add_argument('--save_id',
                        type=int,
                        default=-1)

    parser.add_argument('--use_cuda',
                        action='store_true',
                        help='Whether to use cuda')

    main(parser.parse_args())




