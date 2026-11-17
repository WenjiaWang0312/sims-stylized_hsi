import os
import clip
import torch
import torch.nn as nn


class CLIPExtractor():
    def __init__(self, device, cond_mode='text', clip_version='ViT-B/32', 
                 latent_dim=512, clip_dim=512, max_text_len=20) -> None:
        self.device = device
        self.max_text_len = max_text_len
        self.cond_mode = cond_mode
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)

                
    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device=self.device,
                                                jit=False, download_root=os.environ.get("SIMS_CLIP_CACHE"))
        if str(self.device) == "cpu":
            clip_model.float()
        else:
            clip.model.convert_weights(clip_model)

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = self.device
        max_text_len = self.max_text_len # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()
    
    def encode_text_list(self, raw_text_list):
        return torch.cat([self.encode_text(raw_text) for raw_text in raw_text_list], 0)
