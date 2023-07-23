from two_stream.scripts.VideoSpatialPrediction import VideoSpatialPrediction
from two_stream.scripts.VideoTemporalPrediction import VideoTemporalPrediction
from two_stream.models_res import rgb_resnet101, flow_resnet101
#from models.rgb_resnet import rgb_resnet101
#from models.flow_resnet import flow_resnet101
import os
import shutil
import sys
import torch
import numpy as np
import math
import cv2
import video_transforms
import glob

def estrazione_frame_flow(dizionario, root, video_folder):

    extract_gpu = 'two_stream/extract_gpu'
    cartella_destinazione = 'VideoProcessati'

    if not os.path.isdir(cartella_destinazione):
        print(f'Creazione cartella: {cartella_destinazione}')
        os.makedirs("two_stream/VideoProcessati", exist_ok=True)
    
    for id_persona, video in dizionario.items():
        
        nome_cartella = id_persona

        print(f'\nCreazione cartella: {nome_cartella}')
        path_video = os.path.join(root ,cartella_destinazione, nome_cartella)
        os.makedirs(path_video, exist_ok=True)

        # a dense_flow devo passare il path completo della cartella dove salvare i frame (dense_flow non sta in two-stream-pytorch)
        image_path = f'{os.path.join(path_video)}/img'
        flow_x_path = f'{os.path.join(path_video)}/flow_x'
        flow_y_path = f'{os.path.join(path_video)}/flow_y'

        print(f'{nome_cartella} -> inizio estrazione...')
        print(os.path.join(path_video))
        cmd = f'{extract_gpu} -f={os.path.join(video_folder, video)} -x={flow_x_path} -y={flow_y_path} -i={image_path} -b=20 -t=1 -d=0 -s=1 -o=dir'
        os.system(cmd)
        print(f'{nome_cartella} -> fine estrazione.')

        sys.stdout.flush()

    return True


def softmax(x):
    y = [math.exp(k) for k in x]
    sum_y = math.fsum(y)
    z = [k/sum_y for k in y]

    return z


def average_score(dizionario, root):
    class_file = 'two_stream/datasets/ucf101_splits/classInd.txt'
    f = open(class_file, 'r')
    classi = f.readlines()

    path_pesi_rgb = 'two_stream/model_best_spatial.pth.tar'
    path_pesi_flow = 'two_stream/model_best_temporal.pth.tar'

    params = torch.load(path_pesi_rgb)
    spatial_net = rgb_resnet101(pretrained=False, num_classes=101)
    spatial_net.load_state_dict(params['state_dict'], strict=False)
    spatial_net.cuda()
    spatial_net.eval()
    print(f'\nspatial net ha accuracy {params["best_prec1"]:.2f}%')
 
    params = torch.load(path_pesi_flow)
    temporal_net = flow_resnet101(pretrained=False, num_classes=101)
    temporal_net.load_state_dict(params['state_dict'], strict=False)
    temporal_net.cuda()
    temporal_net.eval()
    print(f'temporal net ha accuracy {params["best_prec1"]:.2f}%')

    result = {}
    for id_persona, video in dizionario.items():
        path = os.path.join(root, 'VideoProcessati', id_persona)
        spatial_prediction = VideoSpatialPrediction(path, spatial_net, 101)
        temporal_prediction = VideoTemporalPrediction(path, temporal_net, 101)

        avg_spatial_pred = np.mean(spatial_prediction, axis=1)
        avg_temporal_pred = np.mean(temporal_prediction, axis=1)

        print('------------------------------------------------------------------------------------')

        print(f'\nvideo {id_persona}')
        # media delle due softmax (nel paper hanno dato più importanza alla spatial 2/3 e 1/3)
        avg_spatial_pred_soft = softmax(avg_spatial_pred)
        pred_index = np.argmax(avg_spatial_pred_soft)
        print(f'spatial_net -> azione \'{classi[pred_index].split()[1]}\' al {(avg_spatial_pred_soft[pred_index]*100):.2f}%')

        avg_temporal_pred_soft = softmax(avg_temporal_pred)
        pred_index = np.argmax(avg_temporal_pred_soft)
        print(f'temporal_net -> azione \'{classi[pred_index].split()[1]}\' al {(avg_temporal_pred_soft[pred_index]*100):.2f}%')

        avg = [(g + h) / 2 for g, h in zip(avg_spatial_pred_soft, avg_temporal_pred_soft)]
        pred_index = np.argmax(avg)
        print(f'average -> azione \'{classi[pred_index].split()[1]}\' al {(avg[pred_index]*100):.2f}%')

        result[id_persona] = classi[pred_index].split()[1]

        # 5 azioni più probabili
        azioni = [classi[i].split()[1] for i in range(101)]
        zippato = zip(avg, azioni)
        sorte = [[val, x] for val, x in sorted(zippato, reverse=True)]
        print('\nazioni più probabili')
        for i in range(5):
            print(f'{sorte[i][1]} al {(sorte[i][0]*100):.2f}%')

    return result


def rec(input_name):
    root = 'two_stream'
    video_folder = f'deep_sort/person_videos/{input_name}'
    #video_folder = "deep_sort/data/input_videos"
    lista = os.listdir(video_folder)
    print(lista)
    dizionario = {lista[i].split('.')[0]: lista[i] for i in range(len(lista))}

    estrazione_frame_flow(dizionario, root, video_folder)
    dizionario_idpersona_azione = average_score(dizionario, root)
    
    print('\nFINE.')
    
    return dizionario_idpersona_azione

if __name__ == '__main__':

    root = '.'
    video_folder = '../deep_sort/person_videos'
    lista = os.listdir(video_folder)
    dizionario = {lista[i].split('.')[0]: lista[i] for i in range(len(lista))}

    # cambia_nomi()
    # sposta_flow()

    estrazione_frame_flow(dizionario, root)
    dizionario_idpersona_azione = average_score(dizionario, root)
    
    print('\nFINE.')