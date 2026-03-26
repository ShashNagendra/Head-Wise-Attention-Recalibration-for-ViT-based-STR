"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention
from modules.vitstr import create_vitstr

import math
from torch.nn import LayerNorm


# [Keep the existing copyright notice and imports]
class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        #print("Debug: Initializing Model")  # Debug print
        
        self.stages = {
            'Trans': opt.Transformation,
            'Feat': opt.FeatureExtraction,
            'Seq': opt.SequenceModeling,
            'Pred': opt.Prediction,
            'ViTSTR': getattr(opt, 'Transformer', False),
            'ViTMAE': getattr(opt, 'TransformerMAE', False)
        }
        
        #print(f"Debug: Stages - {self.stages}")  # Debug print
        
        # Transformation
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), 
                I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        elif opt.Transformation != 'None':
            print('No Transformation module specified')

            
        # Handle transformer models
        if self.stages['ViTSTR'] or self.stages['ViTMAE']:
            #print("Debug: Initializing transformer model")  # Debug print
            model_name = opt.TransformerModel if self.stages['ViTSTR'] else 'vitstr_mae_custom'
            self.vitstr = create_vitstr(
                num_tokens=opt.num_class,
                model=model_name,
                img_size=(opt.imgH, opt.imgW),
                patch_size=(getattr(opt, 'patchH', 16), getattr(opt, 'patchW', 16))
            )
            #print(f"Debug: Transformer model created - {model_name}")  # Debug print
            return 

      

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        
        """ [Rest of the original non-transformer initialization] """

        # Handle both ViTSTR and ViTSTR-MAE cases
        if opt.Transformer or opt.TransformerMAE:
            model_name = opt.TransformerModel if opt.Transformer else 'vitstr_mae_custom'
            vit_kwargs = {
                'img_size': (opt.imgH, opt.imgW),
                'patch_size': (opt.patchH or 4, opt.patchW or 64),  # Default to (4,64) if not specified
                'mask_ratio': opt.mask_ratio if hasattr(opt, 'mask_ratio') else 0.0,
                'use_masking': opt.use_masking if hasattr(opt, 'use_masking') else False
            }
            self.vitstr = create_vitstr(
                num_tokens=opt.num_class, 
                model=model_name,
                **vit_kwargs
            )
            return

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')
    
        


    def forward(self, input, text=None, is_train=True, seqlen=25):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        if self.stages['ViTSTR'] or self.stages['ViTMAE']:
            # Transformer forward pass
            mask_kwargs = {}
            if self.stages['ViTMAE'] and is_train:
                mask_kwargs = {
                    'mask_ratio': getattr(self.opt, 'mask_ratio', 0.75),
                    'max_span_length': getattr(self.opt, 'max_span_length', 3),
                    'use_masking': getattr(self.opt, 'use_masking', False)
                }
            #prediction = self.vitstr(input, seqlen=seqlen, **mask_kwargs)
            # For ViTSTR models that don't support masking, only pass seqlen
            if self.stages['ViTSTR']:
                prediction = self.vitstr(input, seqlen=seqlen)
            else:
                prediction = self.vitstr(input, seqlen=seqlen, **mask_kwargs)
            return prediction       


        """ [Rest of your forward pass for non-transformer models] """
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction
    
class JitModel(Model):
    def __init__(self, opt):
        super(Model, self).__init__()
        vit_kwargs = {
            'img_size': (opt.imgH, opt.imgW),
            'patch_size': (opt.patchH or 4, opt.patchW or 64)
        }
        model_name = opt.TransformerModel if opt.Transformer else 'vitstr_mae_custom'
        self.vitstr = create_vitstr(
            num_tokens=opt.num_class, 
            model=model_name,
            **vit_kwargs
        )

    def forward(self, input, seqlen:int = 27):
        prediction = self.vitstr(input, seqlen=seqlen)
        return prediction