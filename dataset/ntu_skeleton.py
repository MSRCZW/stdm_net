import PyQt5
import pickle
from torch.utils.data import DataLoader, Dataset
# from dataset.video_data import *
# from dataset.skeleton import Skeleton, vis
from dataset.video_data import *
from dataset.skeleton import Skeleton, vis
edge = ((0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5),
        (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 0),
        (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
        (19, 18), (21, 22), (22, 7), (23, 24), (24, 11))


class NTU_SKE(Skeleton):
    def __init__(self, data_path, label_path, window_size, final_size, split,mode='train', decouple_spatial=False,
                 num_skip_frame=None, random_choose=False, center_choose=False,debug = False):
        super().__init__(data_path, label_path, window_size, final_size,split,mode, decouple_spatial, num_skip_frame,random_choose, center_choose,debug = True)
        self.edge = edge
        self.debug = debug

    def load_data(self):
        # print(1)
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load datay

        #super().load_data()
        self.data = np.load(self.data_path, mmap_mode='r')[:, :3]  # NCTVM

        # if self.debug:
        #     print(1111)
        #     self.data = self.data[:100]
        #     self.sample_name,self.label = self.sample_name[:100],self.label[:100]
        #self.data = self.data[:,:,:,index,:]
        #print(self.data.shape)
        #print(self.data)
       # print('22222', self.debug)
       #  if self.debug:
       #      self.label = self.label[0:100]
       #      self.data = self.data[0:100]
       #      self.sample_name = self.sample_name[0:100]
def test(data_path, label_path, vid=None, edge=None, is_3d=False, mode='train'):
    dataset = NTU_SKE(data_path, label_path, window_size=48, final_size=32, mode=mode,
                      random_choose=True, center_choose=False)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    labels = open('../prepare/ntu_120/label.txt', 'r').readlines()
    for i, (data, label) in enumerate(loader):
        if i%1000==0:
            vis(data[0].numpy(), edge=edge, view=1, pause=0.01, title=labels[label.item()].rstrip())

    sample_name = loader.dataset.sample_name
    sample_id = [name.split('.')[0] for name in sample_name]
    index = sample_id.index(vid)
    if mode != 'train':
        data, label, index = loader.dataset[index]
    else:
        data, label = loader.dataset[index]
    # skeleton
    vis(data, edge=edge, view=1, pause=0.1)


if __name__ == '__main__':
    data_path = "../../NTU60/cs_val_data.npy"
    label_path = "../../NTU60/cs_val_label.pkl"
    test(data_path, label_path, vid='S004C001P003R001A032', edge=edge, is_3d=True, mode='train')
