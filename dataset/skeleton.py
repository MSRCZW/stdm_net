import os
import sys
import numpy as np
import pickle
import torch
import random
from torch.utils.data import DataLoader, Dataset
# from dataset.video_data import *
from dataset.video_data import *

class Skeleton(Dataset):
    def __init__(self, data_path, label_path, window_size, final_size,split,
                 mode='train', decouple_spatial=False, num_skip_frame=None,
                 random_choose=False, center_choose=False,debug=False,bone=False,vel=True):
        self.data_path = data_path
        self.label_path = label_path
        self.mode = mode
        self.random_choose = random_choose
        self.center_choose = center_choose
        self.window_size = window_size
        self.final_size = final_size
        self.num_skip_frame = num_skip_frame
        self.decouple_spatial = decouple_spatial
        self.edge = None
        self.debug = debug
        self.split = split
        self.load_data()
        self.bone = bone
        self.vel = vel

        #self.index = [2,3,8,20,4,24,23,11,10,9,22,21,7,6,5,19,18,17,16,0,15,14,14,12,1]
    def load_data(self):
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)
        #with open(self.data_path, 'rb') as f:
        #    data = pickle.load(f)
        #    split, data = data['split'], data['annotations']
        #    identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
        #    split = set(split[self.split])
        #    self.data = [np.transpose(x['keypoint'],(3,1,2,0)) for x in data if x[identifier] in split]
        #    self.label = [x['label'] for x in data if x[identifier] in split]
        #    self.sample_name = [x['frame_dir'] for x in data if x[identifier] in split]
        #print('11111',self.debug)
        # if self.debug:
        #
        #     self.label = self.label[0:100]
        #     self.data = self.data[0:100]
        #     self.sample_name = self.sample_name[0:100]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = self.data[index] 
        label = int(self.label[index])
        sample_name = self.sample_name[index]
        #C, T, V, M = data_numpy.shape
        #if M == 1:
        #    data_numpy = np.concatenate((data_numpy, np.zeros((C, T, V, M))), 3)
        data_numpy = np.array(data_numpy)  # CTVM
        data_numpy = data_numpy[:, data_numpy.sum(0).sum(-1).sum(-1) != 0]  # CTVM
        #valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1)!=0)
        #data_numpy = valid_crop_resize(data_numpy,valid_frame_num,self.p_interval,self.window_size)
        # data transform
        if self.decouple_spatial:
            data_numpy = decouple_spatial(data_numpy, edges=self.edge)
            
        if self.num_skip_frame is not None:
            velocity = decouple_temporal(data_numpy, self.num_skip_frame)
            C, T, V, M = velocity.shape
            data_numpy = np.concatenate((velocity, np.zeros((C, 1, V, M))), 1)
            

        #data_numpy = pad_recurrent_fix(data_numpy, self.window_size)  # if short: pad recurrent
        #data_numpy = uniform_sample_np(data_numpy, self.window_size)  # if long: resize
        if self.random_choose:
            data_numpy = random_sample_np(data_numpy, self.window_size)
            #data_numpy = random_choose_simple(data_numpy, self.final_size)
        else:
            data_numpy = uniform_sample_np(data_numpy, self.window_size)
        if self.center_choose:
            ##data_numpy = uniform_sample_np(data_numpy, self.final_size)
            data_numpy = random_choose_simple(data_numpy, self.final_size, center=True)
        else:
            data_numpy = random_choose_simple(data_numpy, self.final_size)
        if self.bone:
            ntu_pairs = ((1,2),(2,21),(3,21),(4,3),(5,21),(6,5),(7,6),(8,7),(9,21),(10,9),(11,10),(12,11),(13,1),(14,13),(15,14),(16,15),(17,1),(18,17),(19,18),(20,19),(22,23),(21,21),(23,8),(24,25),(25,12))
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1,v2 in ntu_pairs:
                bone_data_numpy[:,:,v1-1]=data_numpy[:,:,v1-1]-data_numpy[:,:,v2-1]
            data_numpy =bone_data_numpy
        if self.vel:
            data_numpy[:,:-1]=data_numpy[:,1:]-data_numpy[:,:-1]
            data_numpy[:,-1]=0
        C,T,V,M = data_numpy.shape
        data_numpy = np.concatenate((data_numpy,np.zeros((C,T,5,M))),2)
        index = [2,3,8,20,4,24,23,11,10,9,22,21,7,6,5,19,18,17,16,28,0,1,25,26,27,29,15,14,13,12]
        data_numpy = data_numpy[:,:,index,:]
        if self.mode == 'train':
            data_numpy = random_rot(data_numpy)
            #data_numpy = random_scale(data_numpy)
            return data_numpy.astype(np.float32), label
        else:

            return data_numpy.astype(np.float32), label ,str(sample_name)

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def vis(data, edge, is_3d=True, pause=0.01, view=0.25, title=''):
    import os

    os.environ['DISPLAY'] = 'localhost:10.0'
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('Qt5Agg')
    C, T, V, M = data.shape

    plt.ion()
    fig = plt.figure()
    if is_3d:
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    ax.set_title(title)
    p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
    import sys
    from os import path
    sys.path.append(
        path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    pose = []
    for m in range(M):
        a = []
        for i in range(len(edge)):
            if is_3d:
                a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
            else:
                a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
        pose.append(a)
    ax.axis([-view, view, -view, view])
    if is_3d:
        ax.set_zlim3d(-view, view)
    for t in range(T):
        for m in range(M):
            for i, (v1, v2) in enumerate(edge):
                x1 = data[:2, t, v1, m]
                x2 = data[:2, t, v2, m]
                if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                    pose[m][i].set_xdata(data[0, t, [v1, v2], m])
                    pose[m][i].set_ydata(data[1, t, [v1, v2], m])
                    if is_3d:
                        pose[m][i].set_3d_properties(data[2, t, [v1, v2], m])
        fig.canvas.draw()
        plt.pause(pause)
    plt.close()
    plt.ioff()

