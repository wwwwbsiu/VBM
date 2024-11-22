import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module,  Conv2d, Parameter,Softmax



#--------------------------------------------------------------------------------------#
""" In the paper for the Position attention module  """
class NewPAM_Module(Module):
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(NewPAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        # 1 q                                                                                                                 
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)  
        # 2 k
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)   # (B,C,N)  

        energy = torch.bmm(proj_query, proj_key)     #(N,N)   
        attention = self.softmax(energy)   
        # 3 v
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)   # (B,C,N)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # (B,C,N)  
        out = out.view(m_batchsize, C, height, width) # (B,C,H,W)

        out = self.gamma*out + x
        return out



#--------------------------------------------------------------------------------------#
"""In the paper for the Channel Attention Module"""
class NewCAM_Module(Module):
    def __init__(self, in_dim):
        super(NewCAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1) # (B,C,N)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1) # (B,N,C)
        energy = torch.bmm(proj_query, proj_key)  #(C,C)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy # (C,C)
        attention = self.softmax(energy_new) 
        proj_value = x.view(m_batchsize, C, -1)  # (B,C,N)

        out = torch.bmm(attention, proj_value)  # (B,C,H,W)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out



class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        a = torch.max(x,1)[0].unsqueeze(1) 
        b = torch.mean(x,1).unsqueeze(1)  
        c = torch.cat((a, b), dim=1)      
        return c




class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x) 
        x_out = self.conv(x_compress) 
        scale = torch.sigmoid_(x_out) 
        return x * scale             



#--------------------------------------------------------------------------------------#
"""  In the paper for the Gate Fusion Unit"""
class gatedFusion(nn.Module):

    def __init__(self, dim):
        super(gatedFusion, self).__init__()
        self.fc1 = nn.Linear(dim, dim, bias=True)
        self.fc2 = nn.Linear(dim, dim, bias=True)

    def forward(self, x1, x2):
        x11 = self.fc1(x1)
        x22 = self.fc2(x2)
        # Generate Weight Representations Through Gated Units
        z = torch.sigmoid(x11+x22)
        # Perform Weighted Summation on Two Input Parts
        out = z*x1 + (1-z)*x2
        return out


    
    
    
#--------------------------------------------------------------------------------------#
"""  In the paper for the CSCA"""    
class CSCA(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(CSCA, self).__init__()
        inter_channels = in_channels // 2
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.sa = NewPAM_Module(inter_channels)
        self.sc = NewCAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        
        self.cw = AttentionGate()
        self.hc = AttentionGate()

        self.gate=gatedFusion(inter_channels)
        self.gamma=nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

        self.fused=nn.Conv2d(out_channels, out_channels, 1)
        #----------------------------------------------------------------#
        
    def forward(self, x):
        feat1 = self.conv5a(x) 
        #---------------------------#
        #  Calculate the spatial correlations
        #---------------------------#
        sa_feat = self.sa(feat1) 
        #---------------------------#
        #  Integrate the relationships between positions
        #---------------------------#
        sa_conv = self.conv51(sa_feat)
        # sa_output = self.conv6(sa_conv)




        feat2 = self.conv5c(x) 
        #---------------------------#
        #  Calculate the relationship between channels
        #---------------------------#
        sc_feat = self.sc(feat2)
        #---------------------------#
        #  Integrate the relationships between channels
        #---------------------------#
        sc_conv = self.conv52(sc_feat) 
        # sc_output = self.conv7(sc_conv)




        #-------------------------------------------------------------#
        #  The interaction between C and W
        #-------------------------------------------------------------#
        x_perm1 = x.permute(0,2,1,3).contiguous() 
        x_out1 = self.cw(x_perm1) 
        x_out11 = x_out1.permute(0,2,1,3).contiguous() 

        
        #--------------------------------------------------------------#
        #  The interaction between C and H
        #--------------------------------------------------------------#
        x_perm2 = x.permute(0,3,2,1).contiguous() 
        x_out2 = self.hc(x_perm2) 
        x_out21 = x_out2.permute(0,3,2,1).contiguous() 
   
        
        


        feat_sum = self.gate(sa_conv.permute(0,2,3,1),sc_conv.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
        
        sasc_output = self.conv8(feat_sum)
        sasc_output=sasc_output+self.gamma*(x_out11+x_out21)
        sasc_output=self.fused(sasc_output)

        return sasc_output
    





