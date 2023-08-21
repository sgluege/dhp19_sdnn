import h5py
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import lava.lib.dl.slayer as slayer
from tqdm import tqdm  # progressbar

class dhp19(Dataset):
    """
    Dataset class for the dhp19 dataset
    """

    def __init__(self, input_tensors, target_tensors, session_tensor, subject_tensor, mov_tensor):
        self.input_tensors = input_tensors
        self.target_tensors = target_tensors
        self.session_tensor = session_tensor
        self.subject_tensor = subject_tensor
        self.mov_tensor = mov_tensor
        
    def __len__(self):
        return len(self.input_tensors)
   
    def __getitem__(self, idx):
        return self.input_tensors[idx].to_dense(), self.target_tensors[idx].to_dense(), self.session_tensor[idx], self.subject_tensor[idx], self.mov_tensor[idx]
        

def plot_sample_from_dataset(input, target, batch_idx, cam_index, time_idx, joint_idx):
    """
    Plot a sample from the dataset
    """
    input_sample = input[batch_idx, 0, :, :, cam_index, time_idx]
    target_sample = target[batch_idx, joint_idx, :, :, cam_index, time_idx]  
    plt_title = f'Input sample from\n batch {batch_idx}, time {time_idx}, cam {cam_index}, target joint {joint_idx}'
    plt.figure()
    plt.imshow(input_sample, cmap='gray')
    plt.imshow(target_sample, alpha=.5)
    plt.title(plt_title)
    plt.show()


def plot_net_output(output, sample_idx, time_idx, joint_idx):
    """
    Plot model output
    """
    plt.figure()
    # arget_sample = target[batch_idx, joint_idx, :, :, cam_index, time_idx]  
    plt_title = f'Output sample from\n sample {sample_idx}, time {time_idx}, target joint {joint_idx}'
    plt.imshow(output[sample_idx, joint_idx,:,:,time_idx])
    plt.title(plt_title)
    plt.show()


def plot_net_input_output(output, input, sample_idx, time_idx, joint_idx):
    """
    Plot model output
    """
    plt.figure()
    # arget_sample = target[batch_idx, joint_idx, :, :, cam_index, time_idx]  
    plt_title = f'Input/Output sample from\n sample {sample_idx}, time {time_idx}, target joint {joint_idx}'
    
    plt.figure()
    plt.imshow(input[sample_idx, 0,:,:,time_idx], cmap='gray')
    plt.imshow(output[sample_idx, joint_idx,:,:,time_idx], alpha=.5)
    plt.title(plt_title)
    plt.title(plt_title)
    plt.show()


# Sparsity loss to penalize the network for high event-rate.
def event_rate_loss(x, max_rate=0.01):
    mean_event_rate = torch.mean(torch.abs(x))
    return F.mse_loss(F.relu(mean_event_rate - max_rate), torch.zeros_like(mean_event_rate))


# Define the network
class Network(torch.nn.Module):
    """
    Define the CNN model as SDNN with slayer blocks
    """

    def __init__(self):
        super(Network, self).__init__()
        
        sdnn_params = { # sigma-delta neuron parameters (taken from PilotNet tutorial)
                'threshold'     : 0.1,    # delta unit threshold
                'tau_grad'      : 0.5,    # delta unit surrogate gradient relaxation parameter
                'scale_grad'    : 1,      # delta unit surrogate gradient scale parameter
                'requires_grad' : True,   # trainable threshold
                'shared_param'  : True,   # layer wise threshold
                'activation'    : F.relu, # activation function
            }
        
        sdnn_cnn_params = { # conv layer has additional mean only batch norm
                **sdnn_params,                                 # copy all sdnn_params
                'norm' : slayer.neuron.norm.MeanOnlyBatchNorm, # mean only quantized batch normalizaton
            }
        # sdnn_dense_params = { # dense layers have additional dropout units enabled
        #         **sdnn_cnn_params,                        # copy all sdnn_cnn_params
        #         'dropout' : slayer.neuron.Dropout(p=0.2), # neuron dropout
        #     }
        
        self.blocks = torch.nn.ModuleList([# sequential network blocks 
                # delta encoding of the input
                slayer.block.sigma_delta.Input(sdnn_params, bias=None), 
                
                # 1. convolution layer 1 stride 1, dilitation 1, max pooling 2x2
                slayer.block.sigma_delta.Conv(sdnn_cnn_params, in_features=1, out_features=16, kernel_size=3, stride=1, padding=1, dilation=1, weight_scale=2, weight_norm=True),
                # torch.Size([1, 16, 260, 344, 16])
                
                # pool 1 mean pooling 2x2 (original architectur uses max pooling)
                slayer.block.sigma_delta.Pool(sdnn_params, kernel_size=2),
                # torch.Size([1, 16, 130, 172, 16])
                
                # 2. convolution layer stride 1, dilitation 1,
                slayer.block.sigma_delta.Conv(sdnn_cnn_params, in_features=16, out_features=32, kernel_size=3, stride=1, padding=1, dilation=1, weight_scale=2, weight_norm=True),
                # torch.Size([1, 32, 130, 172, 16])

                # 3. convolution layer stride 1, dilitation 1,
                slayer.block.sigma_delta.Conv(sdnn_cnn_params, in_features=32, out_features=32, kernel_size=3, stride=1, padding=1, dilation=1, weight_scale=2, weight_norm=True),
                # torch.Size([1, 32, 130, 172, 16])

                # 4. convolution layer stride 1, dilitation 1,
                slayer.block.sigma_delta.Conv(sdnn_cnn_params, in_features=32, out_features=32, kernel_size=3, stride=1, padding=1, dilation=1, weight_scale=2, weight_norm=True),
                # torch.Size([1, 32, 130, 172, 16])
                
                # pool 2 mean pooling 2x2 (original architectur uses max pooling)
                slayer.block.sigma_delta.Pool(sdnn_params, kernel_size=2),
                # torch.Size([1, 32, 65, 86, 16])
                
                # 5. convolution layer stride 1, dilitation 2,
                slayer.block.sigma_delta.Conv(sdnn_cnn_params, in_features=32, out_features=64, kernel_size=3, stride=1, padding=2, dilation=2, weight_scale=2, weight_norm=True),
                # torch.Size([1, 64, 65, 86, 16])
                
                # 6. convolution layer stride 1, dilitation 2,
                slayer.block.sigma_delta.Conv(sdnn_cnn_params, in_features=64, out_features=64, kernel_size=3, stride=1, padding=2, dilation=2, weight_scale=2, weight_norm=True),
                # torch.Size([1, 64, 65, 86, 16])
                
                # 7. convolution layer stride 1, dilitation 2,
                slayer.block.sigma_delta.Conv(sdnn_cnn_params, in_features=64, out_features=64, kernel_size=3, stride=1, padding=2, dilation=2, weight_scale=2, weight_norm=True),
                # torch.Size([1, 64, 65, 86, 16])
                
                # 8. convolution layer stride 1, dilitation 2,
                slayer.block.sigma_delta.Conv(sdnn_cnn_params, in_features=64, out_features=64, kernel_size=3, stride=1, padding=2, dilation=2, weight_scale=2, weight_norm=True),
                # torch.Size([1, 64, 65, 86, 16])
                
                # 9. transposed convolution layer stride 2, dilitation 1, (using a 2x2 kernel instead of 3x3 as reportet in the paper to get correct output dims)
                slayer.block.sigma_delta.ConvT(sdnn_cnn_params, in_features=64, out_features=32, kernel_size=2, stride=2, padding=0, dilation=1, weight_scale=2, weight_norm=True),
                # torch.Size([1, 32, 130, 172, 16])

                # 10. convolution layer stride 1, dilitation 2,
                slayer.block.sigma_delta.Conv(sdnn_cnn_params, in_features=32, out_features=32, kernel_size=3, stride=1, padding=2, dilation=2, weight_scale=2, weight_norm=True),
                # torch.Size([1, 32, 130, 172, 16])
                
                # 11. convolution layer stride 1, dilitation 2,
                slayer.block.sigma_delta.Conv(sdnn_cnn_params, in_features=32, out_features=32, kernel_size=3, stride=1, padding=2, dilation=2, weight_scale=2, weight_norm=True),
                # torch.Size([1, 32, 130, 172, 16])

                # 12. convolution layer stride 1, dilitation 2,
                slayer.block.sigma_delta.Conv(sdnn_cnn_params, in_features=32, out_features=32, kernel_size=3, stride=1, padding=2, dilation=2, weight_scale=2, weight_norm=True),
                # torch.Size([1, 32, 130, 172, 16])

                # 13. convolution layer stride 1, dilitation 2,
                slayer.block.sigma_delta.Conv(sdnn_cnn_params, in_features=32, out_features=32, kernel_size=3, stride=1, padding=2, dilation=2, weight_scale=2, weight_norm=True),
                # torch.Size([1, 32, 130, 172, 16])

                # 14. transposed convolution layer stride 2, dilitation 1, (using a 2x2 kernel instead of 3x3 as reportet in the paper to get correct output dims)
                slayer.block.sigma_delta.ConvT(sdnn_cnn_params, in_features=32, out_features=16, kernel_size=2, stride=2, padding=0, dilation=1, weight_scale=2, weight_norm=True),
                # torch.Size([1, 16, 260, 344, 16])

                # 15. convolution layer stride 1, dilitation 1,
                slayer.block.sigma_delta.Conv(sdnn_cnn_params, in_features=16, out_features=16, kernel_size=3, stride=1, padding=1, dilation=1, weight_scale=2, weight_norm=True),
                # torch.Size([1, 16, 130, 172, 16])

                # 16. convolution layer stride 1, dilitation 1,
                slayer.block.sigma_delta.Conv(sdnn_cnn_params, in_features=16, out_features=16, kernel_size=3, stride=1, padding=1, dilation=1, weight_scale=2, weight_norm=True),
                # torch.Size([1, 16, 130, 172, 16])

                # 17. convolution layer stride 1, dilitation 1, out_features = number of heatmaps to predict
                slayer.block.sigma_delta.Conv(sdnn_cnn_params, in_features=16, out_features=13, kernel_size=3, stride=1, padding=1, dilation=1, weight_scale=2, weight_norm=True)
                # torch.Size([1, 13, 130, 172, 16])
        ])  
        

    def forward(self, x):
        count = []
        event_cost = 0

        for block in self.blocks: 
            # forward computation is as simple as calling the blocks in a loop
            x = block(x)
            if hasattr(block, 'neuron'):
                event_cost += event_rate_loss(x)
                count.append(torch.sum(torch.abs((x[..., 1:]) > 0).to(x.dtype)).item())

        return x, event_cost, torch.FloatTensor(count).reshape((1, -1)).to(x.device)


    def grad_flow(self, path=None):
        # helps monitor the gradient flow
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

        plt.figure()
        plt.semilogy(grad)
        if path:
            plt.savefig(path + 'gradFlow.png')
            plt.close()
        else:
            plt.show()

        return grad

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))


def load_file_(filepath):
    """
    Load data from a .pt file.
    :param filepath: path to the file
    :return: data from the file
    """
    if filepath.endswith('.pt'):
        data = torch.load(filepath)
    else:
        raise ValueError('.pt required format.')
    return data


# path to dhp19 eventframes
event_data_path = './data/'
project_path = './'

# training parameters
batch_size  = 8  # batch size
learning_rate = 0.001 # leaerning rate
lam    = 0.01  # lagrangian for event rate loss
cam_index = 1 # camera index to train on
seq_length = 8 # number of event frames per sequence to be shown to the SDNN

model_name = 'dhp19_sdnn'
device = torch.device('cuda:0')

#load data
data_dict = torch.load(event_data_path + f'dhp19_dataset_sparse_small_seq{seq_length}.pt')
# data_dict['input_tensors'].shape # N T C H W Cam format

# Instantiate Network, Optimizer, Dataset and Dataloader
# define dataset
dhp19_dataset = dhp19(data_dict['input_tensors'],
                        data_dict['target_tensors'],
                        data_dict['session_tensor'],
                        data_dict['subject_tensor'],
                        data_dict['mov_tensor'])

train_loader = DataLoader(dataset=dhp19_dataset, batch_size=8, shuffle=True, num_workers=12)
# dhp19_dataset.__len__()

model = Network().to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

# run training step/minibatch 1
input, target, session, subject, mov = next(iter((train_loader)))

# get mini batch
input = input[:,:,:,:,cam_index,:].to(device) # N C H W T
target = target[:,:,:,:,cam_index,:].to(device) # N C H W T
input.shape # torch.Size([8, 1, 260, 344, 8])
target.shape # torch.Size([8, 13, 260, 344, 8])

# mini batch to SDNN
# zero the parameter gradient
optimizer.zero_grad()
# forward inputs
output, event_cost, count = model(input)

# compute loss + backward + optimize
loss = F.mse_loss(output, target)
loss += lam * event_cost
loss.backward()
optimizer.step()

# plot model input vs output
sample_idx=0
time_idx=0
joint_idx=8
plot_net_output(output.detach().cpu(), sample_idx=sample_idx, time_idx=time_idx, joint_idx=joint_idx)
plot_net_input_output(output.detach().cpu(), input.detach().cpu(), sample_idx=sample_idx, time_idx=time_idx, joint_idx=joint_idx)

# gradient flow
grad = model.grad_flow()
plt.figure()
plt.semilogy(grad)
plt.title('Gradient Flow Step 1')
# plt.savefig(project_path + 'grads_per_layer_step1.png')
# plt.close()
plt.show()

# counts
plt.figure()
plt.plot(count.detach().cpu().numpy().T)
plt.title('Number of events per layer step 1')
# plt.savefig(project_path + 'counts_per_layer_step1.png')
# plt.close()
plt.show()

# run training step/minibatch 2
input, target, session, subject, mov = next(iter((train_loader)))

# get mini batch
input = input[:,:,:,:,cam_index,:].to(device) # N C H W T
target = target[:,:,:,:,cam_index,:].to(device) # N C H W T
input.shape # torch.Size([8, 1, 260, 344, 8])
target.shape # torch.Size([8, 13, 260, 344, 8])

# mini batch to SDNN
# zero the parameter gradient
optimizer.zero_grad()
# forward inputs
output, event_cost, count = model(input)

# compute loss + backward + optimize
loss = F.mse_loss(output, target)
loss += lam * event_cost
loss.backward()
optimizer.step()

# gradient flow
grad = model.grad_flow()
plt.figure()
plt.semilogy(grad)
plt.title('Gradient Flow Step 2')
# plt.savefig(project_path + 'grads_per_layer_step2.png')
# plt.close()
plt.show()

# counts
plt.figure()
plt.plot(count.detach().cpu().numpy().T)
plt.title('Number of events per layer step 2')
plt.savefig(project_path + 'counts_per_layer_step2.png')
plt.close()
plt.show()

   
        
