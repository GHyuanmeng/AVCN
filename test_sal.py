import torch
import torch.nn.functional as F
import time
import os
import sys
import numpy as np
from imageio import imwrite

from utils import AverageMeter

def normalize_data(data):
    data_min = np.min(data)
    data_max = np.max(data)
    data_norm = np.clip((data - data_min) *
                        (255.0 / (data_max - data_min)),
                0, 255).astype(np.uint8)
    return data_norm

def save_video_results(output_buffer, save_path):
    video_outputs = torch.stack(output_buffer)
    for i in range(video_outputs.size(0)):
        save_name = os.path.join(save_path, 'pred_sal_{0:06d}.jpg'.format(i+1))
        imwrite(save_name, normalize_data(video_outputs[i][0].data.numpy()))


test_result = r'/home/your_path/'

def test(data_loader, model, opt):
    print('test')
    all_label_num = 0
    correct_num = 0

    model.eval()

    with torch.no_grad():

        data_time = AverageMeter()

        end_time = time.time()


        for i, (data, valid, targets, gt_da) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            label = gt_da.cuda(non_blocking=True)

            inputs = data['rgb']
            curr_batch_size = inputs.size(0)

            while inputs.size(0) < opt.batch_size:
                inputs = torch.cat((inputs, inputs[0:1, :]), 0)
            while data['audio'].size(0) < opt.batch_size:
                data['audio'] = torch.cat((data['audio'], data['audio'][0:1, :]), 0)

            avc_out = model(inputs, data['audio'])
            out = torch.nn.functional.softmax(avc_out)
            value, position = torch.max(out, 1)

            for idx in range(0, curr_batch_size):
                all_label_num += 1
                if position[idx] == label[idx]:
                    correct_num += 1
                with open(test_result + 'test' + '.txt', 'a') as f:
                    f.write('path:{},item:{},===pre:{},=====label:{}'.format(path, str(frame),
                                                                             position[idx], label[idx]) + '\n')
        with open(test_result + 'test' + '.txt', 'a') as f:
            f.write('correct_num:{},all_num:{},correct_rate:{}'.format(correct_num, str(all_label_num),
                                                                       correct_num / all_label_num) + '\n')
        a = correct_num / all_label_num
        print(a)


