# coding: utf-8

from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
from torch.utils import data
import os
import numpy as np
import random
import h5py
import multiprocessing as mp
import psutil
from PIL import Image
from torch.utils.data import ConcatDataset

use_gpu = torch.cuda.is_available()

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)    # Using multi-GPU
np.random.seed(seed)                # Numpy module
random.seed(seed)                   # Python random module
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class Dataset(data.Dataset):
    def __init__(self, filename_ids, h5, transform, return_label=False):
        'Initialization'
        self.filename_ids = filename_ids
        self.transform = transform
        self.h5 = h5
        self.rl = return_label
        
        transform_list = [
                transforms.ToTensor()
                ]
        self.transform_seg = transforms.Compose(transform_list)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.filename_ids)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        id = self.filename_ids[index]
        fixed_img = self.h5['image']['fixed'][id].astype(np.uint8)
        moving_img = self.h5['image']['warped'][id].astype(np.uint8)
        reference_img = self.h5['image']['moving'][id].astype(np.uint8)
       
        fixed = Image.fromarray(fixed_img).convert('L')
        moving = Image.fromarray(moving_img).convert('L')
        reference = Image.fromarray(reference_img).convert('L')
     
        # Load data and get label
        fixed_image = self.transform(fixed)
        moving_image = self.transform(moving)
        reference_image = self.transform(reference)
        if self.rl:
            fixed_id = self.h5['label']['fixed'][id].astype(np.uint8)
            moving_id = self.h5['label']['moving'][id].astype(np.uint8)
            warped_id = self.h5['label']['warped'][id].astype(np.uint8)
            fixed_id = torch.Tensor(fixed_id).unsqueeze(0)
            moving_id = torch.Tensor(moving_id).unsqueeze(0)
            warped_id = torch.Tensor(warped_id).unsqueeze(0)
            # dvf = torch.Tensor(self.h5['dvf'][id])
            return [fixed_image, reference_image, moving_image], [fixed_id, moving_id, warped_id]
        return [fixed_image, reference_image, moving_image]

class PrepareH5Dataset():
    # prepare for the train and test dataset
    
    def __init__(self, data_path, args):
        self.path = data_path
        self.args = args
    
    def create(self):
        if os.path.isfile(self.path):
            return self.create_data_from_single_set(self.path)
        else:
            filenames = os.listdir(self.path)
            if len(filenames) == 0:
                raise RuntimeError('No dataset in path "{}".'.format(self.path))
            else:
                files = [os.path.join(self.path, file) for file in filenames]
                if len(files) == 1:
                    return self.create_data_from_single_set(files[0])
                else:
                    return self.create_data_from_multi_sets(files)


    def create_data_from_single_set(self, filename):
        f = h5py.File(filename, 'r')
        # filename_ids = list(range(20)) 
        filename_ids = list(range(len(f['image']['fixed']))) 
        
        size = 512
        num_worker = mp.cpu_count()//2 if self.args.worker == 0 else self.args.worker
        print('cpu threads:%d'%(mp.cpu_count()))
        print('cpu cores:%d'%(psutil.cpu_count(False)))
        print('num worker:%d'%num_worker)
        
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(seed)

        partition = {}
        params = {
            'batch_size': self.args.batchsize,
            'shuffle': not self.args.no_shuffle,
            'num_workers': num_worker,
            'worker_init_fn': seed_worker,
            'generator': g,
            'pin_memory': True}

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size, size)),
            transforms.Normalize(0.5, 0.5)])

        partition_train, partition['test'] = train_test_split(
            filename_ids, test_size=self.args.valnum, random_state=seed)
        partition['train'], partition['validation'] = train_test_split(
            partition_train, test_size=self.args.valnum, random_state=seed)

        training_set = Dataset(partition['train'], f, transform, return_label=self.args.label)
        training_generator = data.DataLoader(training_set, **params)
        validation_set = Dataset(partition['validation'], f, transform, return_label=self.args.label)
        validation_generator = data.DataLoader(validation_set, **params)
        test_set = Dataset(partition['test'], f, transform, return_label=self.args.label)
        test_generator = data.DataLoader(test_set, **params)

        return training_generator, validation_generator, test_generator

    def create_data_from_multi_sets(self, filenames):
        filename_ids = list(range(620))

        size = 512
        num_worker = mp.cpu_count()//2 if self.args.worker == 0 else self.args.worker
        print('cpu threads:%d'%(mp.cpu_count()))
        print('cpu cores:%d'%(psutil.cpu_count(False)))
        print('num worker:%d'%num_worker)
        
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(seed)

        partition = {}
        params = {
            'batch_size': self.args.batchsize,
            'shuffle': not self.args.no_shuffle,
            'num_workers': num_worker,
            'worker_init_fn': seed_worker,
            'generator': g,
            'pin_memory': True}

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size, size)),
            transforms.Normalize(0.5, 0.5)])

        partition_train, partition['test'] = train_test_split(
            filename_ids, test_size=self.args.valnum, random_state=seed)
        partition['train'], partition['validation'] = train_test_split(
            partition_train, test_size=self.args.valnum, random_state=seed)
        

        training_set, validation_set, test_set = [], [], []
        for file in filenames:
            f = h5py.File(file, 'r')
            
            training_set.append(Dataset(partition['train'], f, transform, return_label=True))
            validation_set.append(Dataset(partition['validation'], f, transform, return_label=True))
            test_set.append(Dataset(partition['test'], f, transform, return_label=True))

        training_set = ConcatDataset(training_set)
        validation_set = ConcatDataset(validation_set)
        test_set = ConcatDataset(test_set)

        training_generator = data.DataLoader(training_set, **params)
        validation_generator = data.DataLoader(validation_set, **params)
        test_generator = data.DataLoader(test_set, **params)

        return training_generator, validation_generator, test_generator