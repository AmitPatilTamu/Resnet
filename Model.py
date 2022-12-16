import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from NetWork import ResNet
from ImageUtils import parse_record

""" This script defines the training, validation and testing process.
"""

class Cifar(nn.Module):
    def __init__(self, config):
        super(Cifar, self).__init__()
        self.config = config
        self.network = ResNet(
            self.config.resnet_version,
            self.config.resnet_size,
            self.config.num_classes,
            self.config.first_num_filters,
        )
        ### YOUR CODE HERE
        # define cross entropy loss and optimizer
        self.closs = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(self.network.parameters(),lr = 0.01, momentum = 0.9, weight_decay =self.config.weight_decay)

        ### YOUR CODE HERE
    
    def train(self, x_train, y_train, max_epoch):
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size

        print('### Training... ###')
        for epoch in range(1, max_epoch+1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs
            if epoch == 70:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr']/10
            if epoch == 150:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr']/10

            bs = self.config.batch_size
            ### YOUR CODE HERE
            
            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay

                start = i*bs
                end = (i+1) * bs
                batch_x = curr_x_train[start:end,:]
                batch_y = curr_y_train[start:end]
                current_batch = []
                for j in range(self.config.batch_size):
                    current_batch.append(parse_record(batch_x[j],True))
                current_batch = np.array(current_batch)

                tensor_x = torch.cuda.FloatTensor(current_batch)
                tensor_y = torch.cuda.LongTensor(batch_y)

                outputs = self.network(tensor_x)
                loss = self.closs(outputs,tensor_y)
                ### YOUR CODE HERE
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)

            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))

            if epoch % self.config.save_interval == 0:
                self.save(epoch)


    def test_or_validate(self, x, y, checkpoint_num_list):
        self.network.eval()
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(checkpoint_num))
            self.load(checkpointfile)

            preds = []
            for i in tqdm(range(x.shape[0])):
                ### YOUR CODE HERE
                x_pro = parse_record(x[i],False)
                x_input = np.array([x_pro])
                x_tensor = torch.cuda.FloatTensor(x_input)
                pred = self.network(x_tensor)
                max_elements,pred = torch.max(pred,1)
                preds.append(pred)
                ### END CODE HERE

            y = torch.tensor(y)
            preds = torch.tensor(preds)
            #print(preds)
            #print(y)
            print('Test accuracy: {:.4f}'.format(torch.sum(preds==y)/y.shape[0]))
    
    def save(self, epoch):
        checkpoint_path = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(epoch))
        os.makedirs(self.config.modeldir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))