import torch
import torch.nn as nn
import torchvision

class PEN(nn.Module):
    
    def __init__(self, phase, fusetype="mha",phase_1_checkpoint=None, phase_2_checkpoint=None):
        super(PEN, self).__init__()
        
        self.phase = phase
        self.fusetype = fusetype
        
        if self.phase == 1: #Personal branch
            self.lines_backbone = torchvision.models.vit_b_16(weights="DEFAULT")
            self.lines_backbone.conv_proj = torch.nn.Conv2d(5, 768, kernel_size=(16, 16), stride=(16, 16))
            self.lines_backbone.heads = nn.Sequential(
                nn.ReLU(),
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Linear(256,64),
                nn.ReLU(),
                nn.Linear(64,1),
                nn.Sigmoid()
            )
        
        elif self.phase == 2: #Environmental branch
            self.crops_backbone = torchvision.models.vit_b_16(weights="DEFAULT")
            self.crops_backbone.heads = nn.Identity()
            self.crops_mha = nn.MultiheadAttention(768,8,batch_first=True)
            self.crops_fc = nn.Sequential(
                nn.ReLU(),
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Linear(256,64),
                nn.ReLU(),
                nn.Linear(64,1),
                nn.Sigmoid()
            )
            
        elif self.phase == 3: #Fusing
            self.crops_backbone = torchvision.models.vit_b_16(weights="DEFAULT")
            self.crops_backbone.heads = nn.Identity()
            self.crops_mha = nn.MultiheadAttention(768,8,batch_first=True)
            self.crops_fc = nn.Sequential(
                nn.ReLU(),
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Linear(256,64),
                nn.ReLU(),
                nn.Linear(64,1),
                nn.Sigmoid()
            )
            
            self.lines_backbone = torchvision.models.vit_b_16(weights="DEFAULT")
            self.lines_backbone.conv_proj = torch.nn.Conv2d(5, 768, kernel_size=(16, 16), stride=(16, 16))
            self.lines_backbone.heads = nn.Sequential(
                nn.ReLU(),
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Linear(256,64),
                nn.ReLU(),
                nn.Linear(64,1),
                nn.Sigmoid()
            )    
            
            if phase_1_checkpoint != None and phase_2_checkpoint != None: #nottesting
                p1_state_dict = torch.load(phase_1_checkpoint, map_location='cpu')
                p2_state_dict = torch.load(phase_2_checkpoint, map_location='cpu')
                p1_state_dict.update(p2_state_dict)
                self.load_state_dict(p1_state_dict)
            
            if self.fusetype == "mha":
                self.lines_backbone.heads = nn.Identity()
                self.fuse_mha = nn.MultiheadAttention(768,8,batch_first=True)
                self.gelu = nn.GELU()
                self.fuse_fc = nn.Sequential(
                    nn.Linear(768, 256),
                    nn.ReLU(),
                    nn.Linear(256,64),
                    nn.ReLU(),
                    nn.Linear(64,1),
                    nn.Sigmoid()
                )
            if self.fusetype == "add":
                self.lines_backbone.heads = nn.Identity()
                self.fuse_fc = nn.Sequential(
                    nn.Linear(768, 256),
                    nn.ReLU(),
                    nn.Linear(256,64),
                    nn.ReLU(),
                    nn.Linear(64,1),
                    nn.Sigmoid()
                )
            elif self.fusetype == "concat":
                self.lines_backbone.heads = nn.Identity()
                self.fuse_fc = nn.Sequential(
                    nn.Linear(768*2, 256),
                    nn.ReLU(),
                    nn.Linear(256,64),
                    nn.ReLU(),
                    nn.Linear(64,1),
                    nn.Sigmoid()
                )

    def forward(self, input_dict, epoch):
        if self.phase == 1:
            lines_input = torch.cat([input_dict["fp_lines"],input_dict["tp_lines"],input_dict["tp_candidate"]], axis=1)
            lines_output = self.lines_backbone(lines_input)
            
            return lines_output[:,0]
        
        elif self.phase == 2:
            fp_crops = input_dict["fp_crops"]
            batch_size = fp_crops.size(0)
            fp_crops = fp_crops.view(batch_size*5,3,224,224)
            tp_crop = input_dict["tp_crop"][:,0]
            
            if epoch < 4: #Freeze backbone for 3 epochs
                with torch.no_grad():
                    fp_crops = self.crops_backbone(fp_crops)
                    fp_crops = fp_crops.view(batch_size,5,768)
                    tp_crop = self.crops_backbone(tp_crop).unsqueeze(1)
                    
                output = self.crops_mha(tp_crop, fp_crops, fp_crops)[0]
                output = self.crops_fc(output[:,0,:])
            else:
                fp_crops = self.crops_backbone(fp_crops)
                fp_crops = fp_crops.view(batch_size,5,768)
                tp_crop = self.crops_backbone(tp_crop).unsqueeze(1)
                    
                output = self.crops_mha(tp_crop, fp_crops, fp_crops)[0]
                output = self.crops_fc(output[:,0,:])
            return output[:,0]
        
        elif self.phase == 3:
            fp_crops = input_dict["fp_crops"]
            batch_size = fp_crops.size(0)
            fp_crops = fp_crops.view(batch_size*5,3,224,224)
            tp_crop = input_dict["tp_crop"][:,0]
            
            with torch.no_grad():
                fp_crops = self.crops_backbone(fp_crops)
                fp_crops = fp_crops.view(batch_size,5,768)
                tp_crop = self.crops_backbone(tp_crop).unsqueeze(1)

                crops_output = self.crops_mha(tp_crop, fp_crops, fp_crops)[0][:,0,:]
                
                lines_input = torch.cat([input_dict["fp_lines"],input_dict["tp_lines"],input_dict["tp_candidate"]], axis=1)
                lines_output = self.lines_backbone(lines_input)            
                
            if self.fusetype == "mha":
                output = torch.cat([crops_output.unsqueeze(1), lines_output.unsqueeze(1)],axis=1)
                output = self.fuse_mha(output, output, output)[0]
                output = torch.mean(output, axis=1).unsqueeze(1)
                output = self.gelu(output)
                output = self.fuse_fc(output[:,0,:])   
                
            elif self.fusetype == "add":
                output = crops_output+lines_output
                output = self.fuse_fc(output)

            elif self.fusetype == "concat":
                output = torch.cat([crops_output, lines_output],axis=1)
                output = self.fuse_fc(output)
            
            return output[:,0]