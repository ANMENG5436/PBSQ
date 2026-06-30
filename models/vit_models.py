"""ViT-based face recognition models and PBSQ-integrated architectures."""

from .common import *
from .heads import CosFace
from .layers import *
from .patch_utils import extract_patches_pytorch_gridsample
from .comparison_modules import CoattentionModel
from .pbsq import PBSQ

class ViT_face_landmark_patch8_global(nn.Module):
    def __init__(self, *, loss_type, GPU_ID, num_class, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', num_patches=None, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        if num_patches==None:
            num_patches = (image_size // patch_size) ** 2
        # num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # pdb.set_trace()
        self.patch_size = patch_size

        # self.row_num=int(np.sqrt(num_patches)/2)#49
        self.row_num=int(np.sqrt(num_patches))#196
        self.stn=MobileNetV3_backbone(mode='large')
        # # # pdb.set_trace()
        # # # self.stn= ViT_face_stn_patch8(
        # # #                  loss_type = 'None',
        # # #                  GPU_ID = GPU_ID,
        # # #                  num_class = num_class,
        # # #                  image_size=112,
        # # #                  patch_size=8,#8
        # # #                  dim=96,#512
        # # #                  depth=12,#20
        # # #                  heads=3,#8
        # # #                  mlp_dim=1024,
        # # #                  dropout=0.1,
        # # #                  emb_dropout=0.1
        # # #              )
        # # #resnet
        # self.stn=models.resnet50()
        # self.stn.fc=nn.Sequential()
        # # # hybrid_dimension=50
        # # # drop_ratio=0.9

        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.5),    # refer to paper section 6
            nn.Linear(160, self.row_num*self.row_num*2),#2048
        )
        self.global_token = nn.Sequential(
            nn.Dropout(p=0.5),    # refer to paper section 6
            nn.Linear(160, dim),# 
        )
        # self.output_layer = nn.Linear(96, int(self.row_num*self.row_num*2))#49*2,6        mobilenet 96 irse:128
        # self.patch_shape=torch.tensor([2*patch_size,2*patch_size])#49
        self.patch_shape=torch.tensor([patch_size,patch_size])#196
        self.theta=0

        # self.drop_2d=torch.nn.Dropout2d(p=0.1)

        self.dim=dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.sigmoid=nn.Sigmoid()
        self.mlp_head = nn.Sequential(
            # nn.Dropout(p=0.1),
            # nn.Linear(dim,512),
            nn.LayerNorm(dim),#nn.Identity()
        )
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID
        # pdb.set_trace()
        # self.edge_gen=nn.Linear(160, self.row_num*self.row_num*2**2)
        # self.edge_gen=GEM(in_channels=dim,num_classes=num_class)
        # self.gnn=GNN(in_channels=dim,num_classes=145,neighbor_num=15)
        if self.loss_type == 'None':
            print("no loss for vit_face")
        else:
            if self.loss_type == 'Softmax':
                self.loss = Softmax(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
            elif self.loss_type == 'CosFace':
                self.loss = CosFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID,m=0.4)
            elif self.loss_type == 'ArcFace':
                self.loss = ArcFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
            elif self.loss_type == 'SFace':
                self.loss = SFaceLoss(in_features=dim, out_features=num_class, device_id=self.GPU_ID)

    def forward(self, x, image_noaug=None,label= None , mask = None,visualize=False,save_token=False,Random_prob=False,ran_sample=False,glo_diff=False):
        p = self.patch_size
        # x=x/255.0*2-1  #no mean
        # img,fdsa=self.stn(img)
        # pdb.set_trace()
        # x_patch=x.clone()
        # x=x.detach()
        # pdb.set_trace()
        if image_noaug is not None:
            x_aug=x
            x=image_noaug
            
        theta0=self.stn(x)#.forward(x)            #with original stn
        # pdb.set_trace()

        theta0 = theta0.mean(dim=(-2, -1))#average pooling
        theta=self.output_layer(theta0)
        # pdb.set_trace()
        # edge=self.edge_gen(theta0)
        # GCN_out=
        glo_token=self.global_token(theta0).view(-1,1,self.dim)
        
        # #stn
        # # x,asdf=self.stn(x)
        # theta = theta.view(-1, 2, 3)
        # grid = F.affine_grid(theta, x.size())
        # x = F.grid_sample(x, grid)
        #landmark
        # pdb.set_trace()
        #min max scale
        t_max=torch.max(theta,1)[0]#.repeat(1,49*2)
        t_max=torch.unsqueeze(t_max,dim=1).repeat(1,self.row_num*self.row_num*2)
        t_min=torch.min(theta,1)[0]#.repeat(1,49*2)
        t_min=torch.unsqueeze(t_min,dim=1).repeat(1,self.row_num*self.row_num*2)
        theta=(theta-t_min)/(t_max-t_min)*111

        




        # #sigmoid scale
        # theta=self.sigmoid(theta)*111*2-111*0.5
        # theta=(self.sig(theta)*2-0.5)*(self.image_size-1)
        # theta=self.sig(theta)*111
        # theta=theta.round()
        # theta=theta.type(torch.int32)
        # pdb.set_trace()
        theta=theta.view(-1,self.row_num*self.row_num,2)
        self.theta=theta
        theta=theta.detach()
        hbs=np.int64(theta.shape[0]/2)
        # pdb.set_trace()
        if  glo_diff:
            theta0=theta[:hbs]
            theta=theta[hbs:]#.unsqueece(0)
            # theta0=torch.unsqueeze(theta0, 0)
            # theta=torch.unsqueeze(theta, 0)
        if Random_prob:
            # pdb.set_trace()
            # if not glo_diff:
            prob=torch.randn(theta.shape)*2#*12# 10 pixel 
            theta=theta+prob.cuda()
            # else:
            #     prob=torch.randn(theta[1].shape)*3#*12# 10 pixel 
            #     theta[1]=theta[1]+prob.cuda()
            # if not return_prob:
            b,c,fea=theta.shape
            if ran_sample:
                extract_id=torch.randint(0,c,(b,36,1)).cuda()
                keep_num=36
            else:
                extract_id=torch.randint(0,c,(b,self.row_num*self.row_num,1)).cuda()
                keep_num=self.row_num*self.row_num
            extract_id=extract_id.repeat(1,1,2)
            #extract landmarks
            # for i in range(b):
            # extract_id=extract_id.view(592,25,1)
            out_theta=torch.gather(theta, 1, extract_id)
            
            # theta[0][extract_id[0,:,0]][:,1]==out_theta[0][:,1]
            theta=out_theta
            # keep_num=25
            # self.theta=theta#.detach()
            # pdb.set_trace()

            if keep_num is not None:
                num_land=keep_num
            else:
                num_land=theta.shape[-2]#int(self.row_num*self.row_num)
        else:
            num_land=theta.shape[-2]
        #.detach()
        # pdb.set_trace()
        if  glo_diff:
            theta=torch.cat([theta0,theta])
            # theta=theta[1]
        if image_noaug is not None:
            x=extract_patches_pytorch_gridsample(x_aug,theta[:,:num_land],patch_shape=self.patch_shape,num_landm=num_land)
        else:
            x=extract_patches_pytorch_gridsample(x,theta[:,:num_land],patch_shape=self.patch_shape,num_landm=num_land)
        # pdb.set_trace()
        # x=x.detach()
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        # pdb.set_trace()
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        x = torch.cat((glo_token, x), dim=1)
        # x = torch.cat((x,glo_token), dim=1)
        # x=self.drop_2d(x)
        x += self.pos_embedding[:, :(n + 1)]
        # x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, mask)
        # pdb.set_trace()
        # x=self.gnn(x)
        if save_token==True:
            tokens=x[:,1:]
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        # x = x[:,1:,:].mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        emb = self.mlp_head(x)
        
        
        if save_token:
            return emb,tokens,self.theta
        if label is not None:
            x = self.loss(emb, label)
            return x, self.theta
            # return x, emb
        else:
            if visualize==True:
                return emb,self.theta
            else:
                return emb

class ViT_face_landmark_patch8(nn.Module):
    def __init__(self, *, loss_type, GPU_ID, num_class, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls',num_patches=None, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,fp16=True,with_land=False,use_standcoord=False,
    Random_prob=False,shuffle=False):
        super().__init__()
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        if num_patches==None:
            num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # # pdb.set_trace()
        self.patch_size = patch_size
        self.fp16=fp16
        self.num_patches=num_patches
        self.row_num=int(np.sqrt(num_patches)/2)#49
        self.row_num=int(np.sqrt(num_patches))#196
        self.with_land=with_land
        if with_land:
            self.stn=MobileNetV3_backbone(mode='large')
            # # # pdb.set_trace()
            # # # self.stn= ViT_face_stn_patch8(
            # # #                  loss_type = 'None',
            # # #                  GPU_ID = GPU_ID,
            # # #                  num_class = num_class,
            # # #                  image_size=112,
            # # #                  patch_size=8,#8
            # # #                  dim=96,#512
            # # #                  depth=12,#20
            # # #                  heads=3,#8
            # # #                  mlp_dim=1024,
            # # #                  dropout=0.1,
            # # #                  emb_dropout=0.1
            # # #              )
            # # #resnet
            # # self.stn=models.resnet50()
            # # self.stn.fc=nn.Sequential()
            # # # hybrid_dimension=50
            # # # drop_ratio=0.9

            self.output_layer = nn.Sequential(
                nn.Dropout(p=0.5),    # refer to paper section 6
                nn.Linear(160, self.row_num*self.row_num*2),#2048
            )
            
            # self.output_layer = nn.Linear(96, int(self.row_num*self.row_num*2))#49*2,6        mobilenet 96 irse:128
            # self.patch_shape=torch.tensor([2*patch_size,2*patch_size])#49
        self.patch_shape=torch.tensor([patch_size,patch_size])#196
        self.theta=0

        self.drop_2d=torch.nn.Dropout2d(p=0.1)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            # nn.Dropout(p=0.1),
            # nn.Linear(dim,512),
            nn.LayerNorm(dim),#nn.Identity()
        )
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID
        self.sigmoid = nn.Sigmoid()
        self.use_standcoord=use_standcoord
        self.Random_prob=Random_prob
        self.shuffle=shuffle
        if self.use_standcoord:
            range_coor=torch.arange(0,14)*8+4#[0,14]-->[4,12,20,108]
            x,y=torch.meshgrid(range_coor,range_coor)
            self.standard_coord=torch.stack((x,y),2).view(1,-1,2)
        
        if self.loss_type == 'None':
            print("no loss for vit_face")
        else:
            if self.loss_type == 'Softmax':
                self.loss = Softmax(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
            elif self.loss_type == 'CosFace':
                
                # # # pdb.set_trace()
                # from vit_pytorch_my.partial_fc import PartialFCAdamW
                
                # self.loss = PartialFCAdamW(
                #     CosFace_arcimplement(), dim, num_class, 
                #     1.0, self.fp16)
                # # module_partial_fc.train().cuda()
                

                self.loss = CosFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID,m=0.4)
            elif self.loss_type == 'ArcFace':
                self.loss = ArcFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
            elif self.loss_type == 'SFace':
                self.loss = SFaceLoss(in_features=dim, out_features=num_class, device_id=self.GPU_ID)

    def forward(self, x, label= None , mask = None,visualize=False,save_token=False,opt=None,keep_num=None,glo_diff=False):
        p = self.patch_size

        #get the image shape
        b,imgshape=x.shape[0],x.shape[-2]
        if self.num_patches==144 and imgshape==112:
            keep_num=self.num_patches
        else:
            keep_num=(imgshape//p)**2
        # x=x/255.0*2-1  #no mean
        # img,fdsa=self.stn(img)
        # pdb.set_trace()
        # x_patch=x.clone()
        # x=x.detach()
        

        if keep_num is not None:
            num_land=keep_num
        else:
            num_land=int(self.row_num*self.row_num)
        if self.with_land:
            theta=self.stn(x)#.forward(x)            #with original stn
            # pdb.set_trace()

            theta = theta.mean(dim=(-2, -1))#average pooling   for cnn
            theta=self.output_layer(theta)
            
            # #stn
            # # x,asdf=self.stn(x)
            # theta = theta.view(-1, 2, 3)
            # grid = F.affine_grid(theta, x.size())
            # x = F.grid_sample(x, grid)
            #landmark
            # pdb.set_trace()
            #min max scale
            t_max=torch.max(theta,1)[0]#.repeat(1,49*2)
            t_max=torch.unsqueeze(t_max,dim=1).repeat(1,self.row_num*self.row_num*2)
            t_min=torch.min(theta,1)[0]#.repeat(1,49*2)
            t_min=torch.unsqueeze(t_min,dim=1).repeat(1,self.row_num*self.row_num*2)
            theta=(theta-t_min)/(t_max-t_min)*111

            # #sigmoid scale
            # theta=(self.sig(theta)*2-0.5)*(self.image_size-1)
            # theta=self.sigmoid(theta)*111*2-111*0.5
            # theta=theta.round()
            # theta=theta.type(torch.int32)
            theta=theta.view(-1,self.row_num*self.row_num,2)
            self.theta=theta#.detach()
            # pdb.set_trace()

            
            # pdb.set_trace()
            x=extract_patches_pytorch_gridsample(x,theta[:,:num_land],patch_shape=self.patch_shape,num_landm=num_land)
            # pdb.set_trace()
        
        # else:
        #     x=x#.transpose(1,2)
        # pdb.set_trace()
        if self.use_standcoord:
            # if self.use_standcoord:
            range_coor=torch.arange(0,np.sqrt(num_land))*8+4#[0,14]-->[4,12,20,108]
            coor_x,coor_y=torch.meshgrid(range_coor,range_coor)
            standard_coord=torch.stack((coor_x,coor_y),2).view(1,-1,2)
            # pdb.set_trace()
            theta=repeat(standard_coord, '() n d -> b n d', b = b).cuda()#[196,2-->b,196,2]
            b,c,fea=theta.shape
            if self.Random_prob:
                # pdb.set_trace()
                prob=torch.randn(theta.shape)*3#*12# 10 pixel 
                theta=theta+prob.cuda()
                # if not return_prob:
                
                # if ran_sample:
                #     extract_id=torch.randint(0,c,(b,36,1)).cuda()
                #     keep_num=36
                # else:
            if self.shuffle:
                extract_id=torch.randint(0,c,(b,c,1)).cuda()
                # keep_num=self.row_num*self.row_num
                extract_id=extract_id.repeat(1,1,2)
                #extract landmarks
                # for i in range(b):
                # extract_id=extract_id.view(592,25,1)
                out_theta=torch.gather(theta, 1, extract_id)
                theta= out_theta
            # stand_coord=self.standard_coord.repeat()
            # pdb.set_trace()
            # from PIL import Image
            x=extract_patches_pytorch_gridsample(x,theta[:,:num_land],patch_shape=self.patch_shape,num_landm=num_land)
            x=x.permute(0,1,3,2)
            # # x=torch.permute(x,(0,3,1,2))
            # x=x.permute(0,2,3,1)
            # x_gen=x_gen.permute(0,3,2,1)
            
            # # x_gen=torch.permute(x_gen,(0,3,1,2))
            # img = Image.fromarray(((x.cpu().detach().numpy()[0]*0.5+0.5)*255).astype(np.uint8))
            # img.save("faces.png")
            # img = Image.fromarray(((x_gen.cpu().detach().numpy()[0]*0.5+0.5)*255).astype(np.uint8))
            # # img = Image.fromarray(x_gen.cpu().detach().numpy()[0]*0.5+0.5)
            # img.save("faces_gen.png")
        if len(x.shape)==4:
            x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        # pdb.set_trace()
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        # x=self.drop_2d(x)
        x = self.dropout(x)
        x = self.transformer(x, mask)
        if save_token==True:
            tokens=x[:,1:]
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        emb = self.mlp_head(x)
        
        
        if save_token:
            return emb,tokens,self.theta
        if label is not None:
            if opt is not None:
                # pdb.set_trace()
                x = self.loss(emb, label,opt)
                return x, emb
            else:
                x = self.loss(emb, label)
                return x, self.theta
        # elif 

        else:
            if visualize==True:
                return emb,theta#self.theta
            else:
                return emb

class PBSQ_ViT_Part_face(nn.Module):
    def __init__(self, *, loss_type, GPU_ID, num_class, image_size, patch_size, dim, depth, heads, mlp_dim, 
                 pool='cls', num_patches=None, channels=3, dim_head=64, dropout=0., emb_dropout=0., 
                 fp16=True, with_land=False, num_text_queries=3, use_cross_attn=False, **kwargs):
        super().__init__()
        
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        
        if num_patches == None:
            num_patches = (image_size // patch_size) ** 2
        
        patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size
        self.fp16 = fp16
        self.num_patches = num_patches
        self.row_num = int(np.sqrt(num_patches))
        self.with_land = with_land
        self.num_class = num_class
        
        if with_land:
            self.stn = MobileNetV3_backbone(mode='large')
            self.output_layer = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(160, self.row_num * self.row_num * 2),
            )
            self.patch_shape = torch.tensor([patch_size, patch_size])
            self.theta = 0
        
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.dim = dim
        self.pool = pool
        self.to_latent = nn.Identity()
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
        )
        
       
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID
        
        if self.loss_type == 'None':
            print("no loss for vit_face")
        else:
            if self.loss_type == 'Softmax':
                self.loss = Softmax(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
            elif self.loss_type == 'CosFace':
                self.loss = CosFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID, m=0.4)
            elif self.loss_type == 'ArcFace':
                self.loss = ArcFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
            elif self.loss_type == 'SFace':
                self.loss = SFaceLoss(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        

        self.Coattention_module = CoattentionModel(
            all_channel=self.dim,  
            all_dim=int(np.sqrt(num_patches))  
        )
            
    def forward(self, x, label=None, mask=None, visualize=False, save_token=False, opt=None, keep_num=None, **kwargs):
        p = self.patch_size
        
        
        b, imgshape = x.shape[0], x.shape[-2]
        if self.num_patches == 144 and imgshape == 112:
            keep_num = self.num_patches
        else:
            keep_num = (imgshape // p) ** 2
        
        if keep_num is not None:
            num_land = keep_num
        else:
            num_land = int(self.row_num * self.row_num)
        
        if self.with_land:
            theta = self.stn(x)
            theta = theta.mean(dim=(-2, -1))
            theta = self.output_layer(theta)
            
            t_max = torch.max(theta, 1)[0]
            t_max = torch.unsqueeze(t_max, dim=1).repeat(1, self.row_num * self.row_num * 2)
            t_min = torch.min(theta, 1)[0]
            t_min = torch.unsqueeze(t_min, dim=1).repeat(1, self.row_num * self.row_num * 2)
            theta = (theta - t_min) / (t_max - t_min) * 111
            
            theta = theta.view(-1, self.row_num * self.row_num, 2)
            self.theta = theta
            
            x = extract_patches_pytorch_gridsample(x, theta[:, :num_land], 
                                                 patch_shape=self.patch_shape, num_landm=num_land)
        
        
        if len(x.shape) == 4:
            x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        
       
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        
       
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
       
        x = self.transformer(x, mask)
        
        
        x, _ = self.Coattention_module(x)
        
      
        if save_token:
            tokens = x[:, 1:]
        
        
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        
        
        x = self.to_latent(x)
        emb = self.mlp_head(x)
        
        
        if save_token:
            return emb, tokens, self.theta
        
        if label is not None:
            if opt is not None:
                x = self.loss(emb, label, opt)
                return x, emb
            else:
                x = self.loss(emb, label)
                return x, self.theta
        else:
            if visualize:
                return emb, theta if self.with_land else None
            else:
                return emb

class ViT_face_landmark_patch8_4simmin(nn.Module):
    def __init__(self, *, loss_type, GPU_ID, num_class, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls',num_patches=None, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,fp16=True):
        super().__init__()
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        if num_patches==None:
            num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # # pdb.set_trace()
        self.patch_size = patch_size
        self.fp16=fp16
        self.num_patches=num_patches
        self.row_num=int(np.sqrt(num_patches)/2)#49
        self.row_num=int(np.sqrt(num_patches))#196
        # self.stn=MobileNetV3_backbone(mode='large')
        self.dim=dim
        # # # pdb.set_trace()
        # # # self.stn= ViT_face_stn_patch8(
        # # #                  loss_type = 'None',
        # # #                  GPU_ID = GPU_ID,
        # # #                  num_class = num_class,
        # # #                  image_size=112,
        # # #                  patch_size=8,#8
        # # #                  dim=96,#512
        # # #                  depth=12,#20
        # # #                  heads=3,#8
        # # #                  mlp_dim=1024,
        # # #                  dropout=0.1,
        # # #                  emb_dropout=0.1
        # # #              )
        # # #resnet
        # # self.stn=models.resnet50()
        # # self.stn.fc=nn.Sequential()
        # # # hybrid_dimension=50
        # # # drop_ratio=0.9

        # self.output_layer = nn.Sequential(
        #     nn.Dropout(p=0.5),    # refer to paper section 6
        #     nn.Linear(160, self.row_num*self.row_num*2),#2048
        # )
        
        # # self.output_layer = nn.Linear(96, int(self.row_num*self.row_num*2))#49*2,6        mobilenet 96 irse:128
        # # self.patch_shape=torch.tensor([2*patch_size,2*patch_size])#49
        # self.patch_shape=torch.tensor([patch_size,patch_size])#196
        self.theta=0

        self.drop_2d=torch.nn.Dropout2d(p=0.1)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            # nn.Dropout(p=0.1),
            # nn.Linear(dim,512),
            nn.LayerNorm(dim),#nn.Identity()
        )
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID
        self.sigmoid = nn.Sigmoid()
        self.num_features=dim
        self.in_chans=channels


        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self._trunc_normal_(self.mask_token, std=.02)
        # if self.loss_type == 'None':
        #     print("no loss for vit_face")
        # else:
        #     if self.loss_type == 'Softmax':
        #         self.loss = Softmax(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        #     elif self.loss_type == 'CosFace':
                
        #         # # # pdb.set_trace()
        #         # from vit_pytorch_my.partial_fc import PartialFCAdamW
                
        #         # self.loss = PartialFCAdamW(
        #         #     CosFace_arcimplement(), dim, num_class, 
        #         #     1.0, self.fp16)
        #         # # module_partial_fc.train().cuda()
                

        #         self.loss = CosFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID,m=0.4)
        #     elif self.loss_type == 'ArcFace':
        #         self.loss = ArcFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        #     elif self.loss_type == 'SFace':
        #         self.loss = SFaceLoss(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
    def forward(self, x, mask_sim=None,label= None , mask = None,visualize=False,save_token=False,opt=None,keep_num=None):
        p = self.patch_size

        #get the image shape
        imgshape=x.shape[-2]
        if self.num_patches==144 and imgshape==112:
            keep_num=self.num_patches
        else:
            keep_num=(imgshape//p)**2
        # # x=x/255.0*2-1  #no mean
        # # img,fdsa=self.stn(img)
        # # pdb.set_trace()
        # # x_patch=x.clone()
        # # x=x.detach()
        # theta=self.stn(x)#.forward(x)            #with original stn
        # # pdb.set_trace()

        # theta = theta.mean(dim=(-2, -1))#average pooling   for cnn
        # theta=self.output_layer(theta)
        
        # # #stn
        # # # x,asdf=self.stn(x)
        # # theta = theta.view(-1, 2, 3)
        # # grid = F.affine_grid(theta, x.size())
        # # x = F.grid_sample(x, grid)
        # #landmark
        # # pdb.set_trace()
        # # #min max scale
        # # t_max=torch.max(theta,1)[0]#.repeat(1,49*2)
        # # t_max=torch.unsqueeze(t_max,dim=1).repeat(1,self.row_num*self.row_num*2)
        # # t_min=torch.min(theta,1)[0]#.repeat(1,49*2)
        # # t_min=torch.unsqueeze(t_min,dim=1).repeat(1,self.row_num*self.row_num*2)
        # # theta=(theta-t_min)/(t_max-t_min)*111

        # # #sigmoid scale
        # # theta=(self.sig(theta)*2-0.5)*(self.image_size-1)
        # theta=self.sigmoid(theta)*111*2-111*0.5
        # # theta=theta.round()
        # # theta=theta.type(torch.int32)
        # theta=theta.view(-1,self.row_num*self.row_num,2)
        # self.theta=theta#.detach()
        # # pdb.set_trace()

        # if keep_num is not None:
        #     num_land=keep_num
        # else:
        #     num_land=int(self.row_num*self.row_num)
        # # pdb.set_trace()
        # x=extract_patches_pytorch_gridsample(x,theta[:,:num_land],patch_shape=self.patch_shape,num_landm=num_land)
        # pdb.set_trace()
        land_x=x.detach()
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        # pdb.set_trace()
        if mask_sim is not None:
            B, L, _ = x.shape
            mask_token = self.mask_token.expand(B, L, -1)
            w = mask_sim.flatten(1).unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w




        b, n, _ = x.shape
        # pdb.set_trace()
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        # x=self.drop_2d(x)
        x = self.dropout(x)
        x = self.transformer(x, mask)
        # if save_token==True:
        #     tokens=x[:,1:]
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        emb = self.mlp_head(x)[:,1:]#extract visual tokens for reconstruction
        
        
        if save_token:
            return emb,tokens,self.theta
        if label is not None:
            if opt is not None:
                # pdb.set_trace()
                x = self.loss(emb, label,opt)
                return x, emb
            else:
                x = self.loss(emb, label)
                return x, emb
        # elif 

        else:
            if visualize==True:
                return emb,theta#self.theta
            else:
                B, L, C = emb.shape
                H = W = int(L ** 0.5)
                emb = emb.permute(0, 2, 1).reshape(B, C, H, W)
                return emb,land_x

class ViT_face_landmark_patch8_4simmin_glo_loc(nn.Module):
    def __init__(self, *, loss_type, GPU_ID, num_class, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls',num_patches=None, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,fp16=True):
        super().__init__()
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        if num_patches==None:
            num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # # pdb.set_trace()
        self.patch_size = patch_size
        self.fp16=fp16
        self.num_patches=num_patches
        self.row_num=int(np.sqrt(num_patches)/2)#49
        self.row_num=int(np.sqrt(num_patches))#196
        self.stn=MobileNetV3_backbone(mode='large')
        self.dim=dim
        # # # self.stn= ViT_face_stn_patch8(
        # # #                  loss_type = 'None',
        # # #                  GPU_ID = GPU_ID,
        # # #                  num_class = num_class,
        # # #                  image_size=112,
        # # #                  patch_size=8,#8
        # # #                  dim=96,#512
        # # #                  depth=12,#20
        # # #                  heads=3,#8
        # # #                  mlp_dim=1024,
        # # #                  dropout=0.1,
        # # #                  emb_dropout=0.1
        # # #              )
        # # #resnet
        # # self.stn=models.resnet50()
        # # self.stn.fc=nn.Sequential()
        # # # hybrid_dimension=50
        # # # drop_ratio=0.9

        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.5),    # refer to paper section 6
            nn.Linear(160, self.row_num*self.row_num*2),#2048
        )
        self.global_token = nn.Sequential(
            nn.Dropout(p=0.5),    # refer to paper section 6
            nn.Linear(160, dim),# 
        )
        # self.output_layer = nn.Linear(96, int(self.row_num*self.row_num*2))#49*2,6        mobilenet 96 irse:128
        # self.patch_shape=torch.tensor([2*patch_size,2*patch_size])#49
        self.patch_shape=torch.tensor([patch_size,patch_size])#196
        self.theta=0

        self.drop_2d=torch.nn.Dropout2d(p=0.1)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            # nn.Dropout(p=0.1),
            # nn.Linear(dim,512),
            nn.LayerNorm(dim),#nn.Identity()
        )
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID
        self.sigmoid = nn.Sigmoid()
        self.num_features=dim
        self.in_chans=channels


        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self._trunc_normal_(self.mask_token, std=.02)
        # if self.loss_type == 'None':
        #     print("no loss for vit_face")
        # else:
        #     if self.loss_type == 'Softmax':
        #         self.loss = Softmax(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        #     elif self.loss_type == 'CosFace':
                
        #         # # # pdb.set_trace()
        #         # from vit_pytorch_my.partial_fc import PartialFCAdamW
                
        #         # self.loss = PartialFCAdamW(
        #         #     CosFace_arcimplement(), dim, num_class, 
        #         #     1.0, self.fp16)
        #         # # module_partial_fc.train().cuda()
                

        #         self.loss = CosFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID,m=0.4)
        #     elif self.loss_type == 'ArcFace':
        #         self.loss = ArcFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        #     elif self.loss_type == 'SFace':
        #         self.loss = SFaceLoss(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
    
    def forward(self, x, mask_sim=None,label= None , mask = None,visualize=False,save_token=False,opt=None,keep_num=None,knowledge_dis=False):
        p = self.patch_size

        #get the image shape
        imgshape=x.shape[-2]
        if self.num_patches==144 and imgshape==112:
            keep_num=self.num_patches
        elif self.num_patches==196 and imgshape==112:
            keep_num=self.num_patches
        else:
            keep_num=(imgshape//p)**2
        # x=x/255.0*2-1  #no mean
        # img,fdsa=self.stn(img)
        # pdb.set_trace()
        # x_patch=x.clone()
        # x=x.detach()
        theta=self.stn(x)#.forward(x)            #with original stn
        # pdb.set_trace()

        theta0 = theta.mean(dim=(-2, -1))#average pooling   for cnn
        theta=self.output_layer(theta0)
        glo_token=self.global_token(theta0).view(-1,1,self.dim)
        
        # #stn
        # # x,asdf=self.stn(x)
        # theta = theta.view(-1, 2, 3)
        # grid = F.affine_grid(theta, x.size())
        # x = F.grid_sample(x, grid)
        #landmark
        # pdb.set_trace()
        #min max scale
        t_max=torch.max(theta,1)[0]#.repeat(1,49*2)
        t_max=torch.unsqueeze(t_max,dim=1).repeat(1,self.row_num*self.row_num*2)
        t_min=torch.min(theta,1)[0]#.repeat(1,49*2)
        t_min=torch.unsqueeze(t_min,dim=1).repeat(1,self.row_num*self.row_num*2)
        theta=(theta-t_min)/(t_max-t_min)*111

        # #sigmoid scale
        # theta=(self.sig(theta)*2-0.5)*(self.image_size-1)
        # theta=self.sigmoid(theta)*111*2-111*0.5
        # theta=theta.round()
        # theta=theta.type(torch.int32)
        theta=theta.view(-1,self.row_num*self.row_num,2)
        self.theta=theta#.detach()
        # pdb.set_trace()

        if keep_num is not None:
            num_land=keep_num
        else:
            num_land=int(self.row_num*self.row_num)
        # pdb.set_trace()
        x=extract_patches_pytorch_gridsample(x,theta[:,:num_land],patch_shape=self.patch_shape,num_landm=num_land)
        # pdb.set_trace()
        
        land_x=x.detach()
        x=x.detach()
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        # pdb.set_trace()
        if mask_sim is not None:
            B, L, _ = x.shape
            mask_token = self.mask_token.expand(B, L, -1)
            w = mask_sim.flatten(1).unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w




        b, n, _ = x.shape
        # pdb.set_trace()
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        x = torch.cat((glo_token, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        # x=self.drop_2d(x)
        x = self.dropout(x)
        x = self.transformer(x, mask)
        # if save_token==True:
        #     tokens=x[:,1:]
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        emb = self.mlp_head(x)[:,1:]#extract visual tokens for reconstruction
        glo_out=self.mlp_head(x)[:,0]
        # pdb.set_trace()
        if save_token:
            return emb,tokens,self.theta
        if label is not None:
            if opt is not None:
                # pdb.set_trace()
                x = self.loss(emb, label,opt)
                return x, emb
            else:
                x = self.loss(emb, label)
                return x, emb,
        # elif 

        else:
            if visualize==True:
                return emb,theta#self.theta
            else:
                
                B, L, C = emb.shape
                H = W = int(L ** 0.5)
                emb = emb.permute(0, 2, 1).reshape(B, C, H, W)
                if knowledge_dis:
                    return emb,land_x,glo_out,theta
                else:
                    return emb,land_x,glo_out

class mobile_dino(nn.Module):
    def __init__(self, embd_dim=128):
        super().__init__()
        self.stn=MobileNetV3_backbone(mode='large')
        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.5),    # refer to paper section 6
            nn.Linear(160, embd_dim),#2048
        )
    def forward(self,x):
        theta=self.stn(x)#.forward(x)            #with original stn
        # pdb.set_trace()

        theta0 = theta.mean(dim=(-2, -1))#average pooling   for cnn
        theta=self.output_layer(theta0)
        return theta

class face_landmark_4simmin_glo_loc(nn.Module):
    def __init__(self, *, loss_type, GPU_ID, num_class, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls',num_patches=None, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,fp16=True):
        super().__init__()
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        if num_patches==None:
            num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # # pdb.set_trace()
        self.patch_size = patch_size
        self.fp16=fp16
        self.num_patches=num_patches
        self.row_num=int(np.sqrt(num_patches)/2)#49
        self.row_num=int(np.sqrt(num_patches))#196
        self.stn=MobileNetV3_backbone(mode='large')
        self.dim=dim
        # # # self.stn= ViT_face_stn_patch8(
        # # #                  loss_type = 'None',
        # # #                  GPU_ID = GPU_ID,
        # # #                  num_class = num_class,
        # # #                  image_size=112,
        # # #                  patch_size=8,#8
        # # #                  dim=96,#512
        # # #                  depth=12,#20
        # # #                  heads=3,#8
        # # #                  mlp_dim=1024,
        # # #                  dropout=0.1,
        # # #                  emb_dropout=0.1
        # # #              )
        # # #resnet
        # # self.stn=models.resnet50()
        # # self.stn.fc=nn.Sequential()
        # # # hybrid_dimension=50
        # # # drop_ratio=0.9

        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.5),    # refer to paper section 6
            nn.Linear(160, self.row_num*self.row_num*2),#2048
        )
        self.global_token = nn.Sequential(
            nn.Dropout(p=0.5),    # refer to paper section 6
            nn.Linear(160, dim),# 
        )
        # self.output_layer = nn.Linear(96, int(self.row_num*self.row_num*2))#49*2,6        mobilenet 96 irse:128
        # self.patch_shape=torch.tensor([2*patch_size,2*patch_size])#49
        self.patch_shape=torch.tensor([patch_size,patch_size])#196
        self.theta=0

        self.drop_2d=torch.nn.Dropout2d(p=0.1)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # self.pool = pool
        # self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(
        #     # nn.Dropout(p=0.1),
        #     # nn.Linear(dim,512),
        #     nn.LayerNorm(dim),#nn.Identity()
        # )
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID
        self.sigmoid = nn.Sigmoid()
        self.num_features=dim
        self.in_chans=channels


        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self._trunc_normal_(self.mask_token, std=.02)
        # if self.loss_type == 'None':
        #     print("no loss for vit_face")
        # else:
        #     if self.loss_type == 'Softmax':
        #         self.loss = Softmax(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        #     elif self.loss_type == 'CosFace':
                
        #         # # # pdb.set_trace()
        #         # from vit_pytorch_my.partial_fc import PartialFCAdamW
                
        #         # self.loss = PartialFCAdamW(
        #         #     CosFace_arcimplement(), dim, num_class, 
        #         #     1.0, self.fp16)
        #         # # module_partial_fc.train().cuda()
                

        #         self.loss = CosFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID,m=0.4)
        #     elif self.loss_type == 'ArcFace':
        #         self.loss = ArcFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        #     elif self.loss_type == 'SFace':
        #         self.loss = SFaceLoss(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
    def forward(self, x, x_Aug=None,keep_num=None,patch_shape=torch.tensor([10,10]),Random_prob=False,return_prob=False,ran_sample=False,random_coor=False,return_land=False):
        p = self.patch_size
        if not random_coor:
            #get the image shape
            imgshape=x.shape[-2]
            if self.num_patches==144 and imgshape==112:
                keep_num=self.num_patches
            elif self.num_patches==196 and imgshape==112:
                keep_num=self.num_patches
            else:
                keep_num=(imgshape//p)**2
            # x=x/255.0*2-1  #no mean
            # img,fdsa=self.stn(img)
            # pdb.set_trace()
            # x_patch=x.clone()
            # x=x.detach()
            theta=self.stn(x)#.forward(x)            #with original stn
            # pdb.set_trace()

            theta0 = theta.mean(dim=(-2, -1))#average pooling   for cnn
            theta=self.output_layer(theta0)
            glo_token=self.global_token(theta0).view(-1,1,self.dim)
            
            # #stn
            # # x,asdf=self.stn(x)
            # theta = theta.view(-1, 2, 3)
            # grid = F.affine_grid(theta, x.size())
            # x = F.grid_sample(x, grid)
            #landmark
            # pdb.set_trace()
            #min max scale
            t_max=torch.max(theta,1)[0]#.repeat(1,49*2)
            t_max=torch.unsqueeze(t_max,dim=1).repeat(1,self.row_num*self.row_num*2)
            t_min=torch.min(theta,1)[0]#.repeat(1,49*2)
            t_min=torch.unsqueeze(t_min,dim=1).repeat(1,self.row_num*self.row_num*2)
            theta=(theta-t_min)/(t_max-t_min)*111

            # #sigmoid scale
            # theta=(self.sig(theta)*2-0.5)*(self.image_size-1)
            # theta=self.sigmoid(theta)*111*2-111*0.5
            # theta=theta.round()
            # theta=theta.type(torch.int32)
            theta=theta.view(-1,self.row_num*self.row_num,2)
            if Random_prob:
                # pdb.set_trace()
                prob=torch.randn(theta.shape)*5#*12# 10 pixel 
                theta=theta+prob.cuda()
                if not return_prob:
                    b,c,fea=theta.shape
                    if ran_sample:
                        extract_id=torch.randint(0,c,(b,36,1)).cuda()
                        keep_num=36
                    else:
                        extract_id=torch.randint(0,c,(b,self.row_num*self.row_num,1)).cuda()
                        keep_num=self.row_num*self.row_num
                    extract_id=extract_id.repeat(1,1,2)
                    #extract landmarks
                    # for i in range(b):
                    # extract_id=extract_id.view(592,25,1)
                    out_theta=torch.gather(theta, 1, extract_id)
                    
                    # theta[0][extract_id[0,:,0]][:,1]==out_theta[0][:,1]
                    theta=out_theta
                # keep_num=25
            self.theta=theta#.detach()
            # pdb.set_trace()

            if keep_num is not None:
                num_land=keep_num
            else:
                num_land=theta.shape[-2]#int(self.row_num*self.row_num)
            # pdb.set_trace()
            # if return_land_only:
            #     return theta,0
            # else:
        else:
            # coor_max=theta.max()#get max value
            # pdb.set_trace()
            b=x.shape[0]
            if not ran_sample:
                num_land=self.row_num*self.row_num
            else:
                num_land=25
            new_theta=torch.rand(b,num_land,2)*111.0#.cuda()
            theta=new_theta.cuda()
        # pdb.set_trace()
        if return_land:
            return theta,x
        else:
            if x_Aug is None:
                x=extract_patches_pytorch_gridsample(x,theta[:,:num_land],patch_shape=self.patch_shape,num_landm=num_land)
            else:
                x=extract_patches_pytorch_gridsample(x_Aug,theta[:,:num_land],patch_shape=self.patch_shape,num_landm=num_land)
            return theta,x

class ViTs_face_overlap_4simmim(nn.Module):
    def __init__(self, *, loss_type, GPU_ID, num_class, image_size, patch_size, ac_patch_size,
                         pad, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * ac_patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # pdb.set_trace()
        self.patch_size = patch_size
        self.soft_split = nn.Unfold(kernel_size=(ac_patch_size, ac_patch_size), stride=(self.patch_size, self.patch_size), padding=(pad, pad))


        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
        )
        self.loss_type = loss_type
        self.pred=None
        self.GPU_ID = GPU_ID
        self.fc=None
        self.num_features=dim
        self.in_chans=channels
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self._trunc_normal_(self.mask_token, std=.02)
        # if self.loss_type == 'None':
        #     print("no loss for vit_face")
        # else:
        #     if self.loss_type == 'Softmax':
        #         self.loss = Softmax(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        #     elif self.loss_type == 'CosFace':
        #         self.loss = CosFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        #     elif self.loss_type == 'ArcFace':
        #         self.loss = ArcFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        #     elif self.loss_type == 'SFace':
        #         self.loss = SFaceLoss(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, img, mask_sim=None,label= None , mask = None,visualize=False,save_token=False,opt=None,keep_num=None,knowledge_dis=False):
        p = self.patch_size
        # pdb.set_trace()
        land_x=img.detach()
        x = self.soft_split(img).transpose(1, 2)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape


        if mask_sim is not None:
            B, L, _ = x.shape
            mask_token = self.mask_token.expand(B, L, -1)
            w = mask_sim.flatten(1).unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]


        x = self.dropout(x)
        x = self.transformer(x, mask)

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        emb = self.mlp_head(x)[:,1:]#extract visual tokens for reconstruction
        glo_out=self.mlp_head(x)[:,0]
        # emb = self.mlp_head(x)
        # pdb.set_trace()


        if label is not None:
            x = self.loss(emb, label)
            return x, emb
        # else:
        else:
            if visualize==True:
                return emb,theta#self.theta
            else:
                B, L, C = emb.shape
                H = W = int(L ** 0.5)
                emb = emb.permute(0, 2, 1).res
