"""Patch extraction and small utility helpers."""

from .common import *

def bn_init(bn):
    bn.weight.data.fill_(1)
    bn.bias.data.zero_()
def create_e_matrix(n):
    end = torch.zeros((n*n,n))
    for i in range(n):
        end[i * n:(i + 1) * n, i] = 1
    start = torch.zeros(n, n)
    for i in range(n):
        start[i, i] = 1
    start = start.repeat(n,1)
    return start,end

def extract_patches_pytorch_gridsample(imgs, landmarks, patch_shape,num_landm=49):#numpy
    """ Extracts patches from an image.
    Args:
        imgs: a numpy array of dimensions [batch_size, width, height, channels]
        landmarks: a numpy array of dimensions [num_patches, 2]
        patch_shape: (width, height)
    Returns:
        a numpy array [num_patches, width, height, channels]
    """
    # pdb.set_trace()
    device=landmarks.device
    # imgs=imgs.to(device)
    # patch_shape = np.array(patch_shape)
    # patch_shape = np.array(patch_shape)
    # patch_half_shape = torch.require(torch.round(patch_shape / 2), dtype=int)
    img_shape=imgs.shape[2]
    # pdb.set_trace()
    list_patches = []
    patch_half_shape=patch_shape/2
    start = -patch_half_shape
    end = patch_half_shape
    # sampling_grid = torch.meshgrid[start[0]:end[0], start[1]:end[1]]
    sampling_grid = torch.meshgrid(torch.arange(start[0],end[0]),torch.arange(start[1],end[1]))#         start[0]:end[0], start[1]:end[1]]
    sampling_grid=torch.stack(sampling_grid,dim=0).to(device)#.cuda()
    # sampling_grid = sampling_grid.swapaxes(0, 2).swapaxes(0, 1)
    sampling_grid=torch.transpose(torch.transpose(sampling_grid,0,2),0,1)
    for i in range(num_landm):
        
        land=landmarks[:,i,:]

        patch_grid = (sampling_grid[None, :, :, :] + land[:, None, None, :])/(img_shape*0.5)-1
        sing_land_patch= F.grid_sample(imgs, patch_grid,align_corners=False)
        list_patches.append(sing_land_patch)
    # pdb.set_trace()
    list_patches=torch.stack(list_patches,dim=2)#.shape
    B, c, patches_num,w,h = list_patches.shape
    row=int(np.sqrt(patches_num))
    list_patches=list_patches.reshape(B,c,row,row,w,h)
    list_patches=list_patches.permute(0,1,2,4,3,5)
    list_patches=list_patches.reshape(B,c,w*int(np.sqrt(patches_num)),h*int(np.sqrt(patches_num)))
                      #.astype('int32')
    return list_patches#list_patches.cuda()

