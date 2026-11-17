import torch
import torch.nn as nn
import numpy as np
loss_mse = nn.MSELoss()
cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, input_feats, 
                 latent_dim=512, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1,
                 activation="gelu"):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        
        self.input_feats = input_feats

        self.muQuery = nn.Parameter(torch.randn(1, self.latent_dim))
        self.sigmaQuery = nn.Parameter(torch.randn(1, self.latent_dim))
        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)

    def forward(self, x, mask): # bs, nfeats, nframes; bs, nframes
        bs, nfeats, nframes = x.shape
        x = x.permute((2, 0, 1)).reshape(nframes, bs, nfeats)

        # embedding of the skeleton
        xseq = self.skelEmbedding(x)

        # add positional encoding
        xseq = self.sequence_pos_encoder(xseq)

        # create a bigger mask, to allow attend to mu and sigma
        # muandsigmaMask = torch.ones((bs, 2), dtype=bool, device=x.device)

        # maskseq = torch.cat((muandsigmaMask, mask), axis=1)

        final = self.seqTransEncoder(xseq, src_key_padding_mask=~mask)
        mu = final[0]
        # logvar = final[1]

        return mu


class TransformerDecoder(nn.Module):
    def __init__(self, input_feats,
                 latent_dim=512, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu",
                ):
        super().__init__()

        self.latent_dim = latent_dim
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
                
        self.input_feats = input_feats

        self.actionBiases = nn.Parameter(torch.randn(1, self.latent_dim))

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)
        
        self.finallayer = nn.Linear(self.latent_dim, self.input_feats)
        
    def forward(self, z, mask):
        latent_dim = z.shape[1]
        bs, nframes = mask.shape

        z = z[None]  # sequence of size 1  #

        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        
        timequeries = self.sequence_pos_encoder(timequeries)
        
        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        
        output = self.finallayer(output) # bs, nframes, nfeats
        
        # zero for padded area
        output[~mask.T] = 0
        output = output.permute((1, 2, 0))

        return output
    

class MotionCLIP(nn.Module):
    def __init__(self, input_feats=125, loss_weight=None, clip_dim=512, latent_dim=64, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu", **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.loss_weight = loss_weight
        self.encoder = TransformerEncoder(input_feats, latent_dim, ff_size, num_layers, num_heads, dropout, activation)
        self.decoder = TransformerDecoder(input_feats, latent_dim, ff_size, num_layers, num_heads, dropout, activation)
        self.clip_downsample = nn.Sequential(nn.Linear(clip_dim, 256), nn.ReLU(), nn.Linear(256, latent_dim))
        self.clip_upsample = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, clip_dim))

    def compute_loss(self, input_motion, pred_motion, pred_z, clip_features, mask, clip_f_ds, clip_f_recon, extra_clip_features):
        # compute all losses other than clip
        total_loss = 0.
        loss_dict = {}
        recon_loss = (input_motion - pred_motion).pow(2).sum(1) * mask
        recon_loss = recon_loss.sum() / mask.sum()
        loss_dict['recon_loss'] = recon_loss.item()
        # compute clip losses
        mixed_clip_loss, clip_loss_dict = self.compute_clip_losses(pred_z, clip_features, clip_f_ds, clip_f_recon, extra_clip_features)
        loss_dict.update(clip_loss_dict)
        # mix and add clip losses
        total_loss = recon_loss * self.loss_weight['recon'] + mixed_clip_loss  # this is the ultimate loss to optimize, 

        return total_loss, loss_dict

    def forward(self, x, mask, clip_feature): # bs, input_feats, nframes; bs, nframes
        mu = self.encoder(x, mask) # bs, latent_dim
        mu = mu / mu.norm(dim=-1, keepdim=True)
        output = self.decoder(mu, mask) # bs, nfeats, nframes
        clip_feature_ds = self.encode_text(clip_feature) # bs, nframes, latent_dim
        clip_feature_recon = self.clip_upsample(clip_feature_ds) # bs, nframes, clip_dim
        return output, mu, clip_feature_ds, clip_feature_recon
    
    def encode(self, x): # bs, input_feats, nframes;
        mask = torch.ones(x.shape[0], x.shape[2], dtype=bool, device=x.device)
        mu = self.encoder(x, mask)
        mu = mu / mu.norm(dim=-1, keepdim=True)
        return mu
    
    def encode_text(self, clip_feature): # bs, input_feats, nframes;
        clip_feature_ds = self.clip_downsample(clip_feature) # bs, nframes, latent_dim
        clip_feature_ds = clip_feature_ds / clip_feature_ds.norm(dim=-1, keepdim=True)
        return clip_feature_ds
    
    def generate(self, mu, mask):
        output = self.decoder(mu, mask)
        return output
    
    def compute_clip_losses(self, motion_features, clip_features, clip_f_ds, clip_f_recon, extra_clip_features):
        total_loss = 0.
        loss_dict = {}
        # normalized features
        clip_features_norm = clip_features / clip_features.norm(dim=-1, keepdim=True)
        clip_f_recon_norm = clip_f_recon / clip_f_recon.norm(dim=-1, keepdim=True)
        
        extra_clip_embeding = self.encode_text(extra_clip_features)

        # if 'mse' in self.loss_weight:
        #     motion_clip_mse_loss = loss_mse(clip_f_ds, motion_features)
        #     loss_dict[f'motion_clip_mse_loss'] = motion_clip_mse_loss.item()
        #     total_loss += motion_clip_mse_loss * self.loss_weight['mse']

        if self.loss_weight.get('cosine', 0) > 0:
            cos = cosine_sim(clip_f_ds, motion_features)
            motion_clip_cosine_loss = (1 - cos).mean()
            loss_dict['motion_clip_cosine_loss'] = motion_clip_cosine_loss.item()
            total_loss += motion_clip_cosine_loss * self.loss_weight['cosine']

        if self.loss_weight.get('clip_mse', 0) > 0:
            clip_recon_mse_loss = loss_mse(clip_features, clip_f_recon)
            loss_dict[f'clip_recon_mse_loss'] = clip_recon_mse_loss.item()
            total_loss += clip_recon_mse_loss * self.loss_weight['clip_mse']

        if self.loss_weight.get('clip_cosine', 0) > 0:
            cos = cosine_sim(clip_f_ds, extra_clip_embeding)
            clip_recon_cosine_loss = (1 - cos).mean()
            loss_dict['clip_recon_cosine_loss'] = clip_recon_cosine_loss.item()
            total_loss += clip_recon_cosine_loss * self.loss_weight['clip_cosine']
            
        return total_loss, loss_dict

