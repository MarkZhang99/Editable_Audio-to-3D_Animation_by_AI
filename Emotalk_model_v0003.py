import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from utils import init_biased_mask, enc_dec_mask
from torch.utils.checkpoint import checkpoint

class EmoTalk(nn.Module):
    def __init__(self, args):
        super(EmoTalk, self).__init__()
        self.args = args
        self.device = args.device

        # === Content Feature Extractor ===
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        )
        self.extractor = Wav2Vec2Model.from_pretrained(
            "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        ).to(self.device)
        # Freeze and disable grad to save memory
        self.extractor.eval()
        for p in self.extractor.parameters():
            p.requires_grad = False
        # Enable HF encoder checkpointing
        if hasattr(self.extractor, 'gradient_checkpointing_enable'):
            self.extractor.gradient_checkpointing_enable()
        self.extractor_proj = nn.Linear(1024, args.transformer_dim).to(self.device)

        # === Emotion Feature Extractor ===
        self.emo_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "r-f/wav2vec-english-speech-emotion-recognition"
        )
        self.emo_extractor = Wav2Vec2Model.from_pretrained(
            "r-f/wav2vec-english-speech-emotion-recognition"
        ).to(self.device)
        # Freeze and disable grad to save memory
        self.emo_extractor.eval()
        for p in self.emo_extractor.parameters():
            p.requires_grad = False
        # Enable HF emotion extractor checkpointing
        if hasattr(self.emo_extractor, 'gradient_checkpointing_enable'):
            self.emo_extractor.gradient_checkpointing_enable()
        self.audio_feature_map_emo = nn.Linear(1024, args.transformer_dim).to(self.device)
        self.emo_gru = nn.GRU(
            input_size=args.transformer_dim,
            hidden_size=args.emo_gru_hidden,
            num_layers=args.emo_gru_layers,
            bidirectional=True,
            batch_first=True
        ).to(self.device)

        # === FiLM Parameters ===
        self.film_gamma = nn.Linear(args.emo_gru_hidden * 2, args.transformer_dim).to(self.device)
        self.film_beta  = nn.Linear(args.emo_gru_hidden * 2, args.transformer_dim).to(self.device)

        # === Level + Person Embedding ===
        self.obj_vector_level  = nn.Linear(args.num_emotions, 32).to(self.device)
        self.obj_vector_person = nn.Embedding(args.num_person, 32).to(self.device)

        # === Transformer Decoder ===
        decoder_input_dim = args.transformer_dim + 64
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_input_dim,
            nhead=args.transformer_heads,
            dim_feedforward=decoder_input_dim * 4,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.transformer_layers).to(self.device)
        self.linear  = nn.Linear(decoder_input_dim, args.bs_dim).to(self.device)

        # === Attention Mask ===
        self.biased_mask1 = init_biased_mask(
            n_head=args.transformer_heads,
            max_seq_len=args.max_seq_len,
            period=args.period
        ).to(self.device)

        # === Memory Projection for Decoder Matching ===
        self.emo_memory_proj = nn.Linear(
            args.emo_gru_hidden * 2,
            decoder_input_dim
        ).to(self.device)

        # === Aliases for backward compatibility ===
        self.audio_encoder_cont    = self.extractor
        self.audio_feature_map_cont = self.extractor_proj
        self.transformer_decoder   = self.decoder
        self.bs_map_r              = self.linear

    def apply_film(self, content_feat, emotion_feat):
        gamma = self.film_gamma(emotion_feat)
        beta  = self.film_beta(emotion_feat)
        return gamma * content_feat + beta

    def forward(self, audio, level, person):
        # === Adjust labels from 1..N to 0..N-1 ===
        if level.min() >= 1 and level.max() <= self.args.num_emotions:
            level = level - 1
        else:
            raise ValueError(f"Level labels must be in [1, {self.args.num_emotions}], got {level}")
        if person.min() >= 1 and person.max() <= self.args.num_person:
            person = person - 1
        else:
            raise ValueError(f"Person labels must be in [1, {self.args.num_person}], got {person}")

        # === Preprocess audio ===
        if audio.dim() == 4:
            audio = audio.squeeze(1).squeeze(1)
        elif audio.dim() == 3:
            audio = audio.squeeze(1)
        elif audio.dim() != 2:
            raise ValueError(f"Unsupported audio shape: {audio.shape}")

        # === Content Features (no grad) ===
        with torch.no_grad():
            input_values = self.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt"
            )['input_values'].to(self.device).squeeze(1)
            cont_feat = self.extractor(input_values).last_hidden_state
        cont_feat = cont_feat.detach()
        cont_feat = self.extractor_proj(cont_feat)

        # === Emotion Features (no grad) ===
        with torch.no_grad():
            emo_input = self.emo_feature_extractor(
                audio, sampling_rate=16000, return_tensors='pt'
            )['input_values'].to(self.device).squeeze(1)
            emo_hidden = self.emo_extractor(emo_input).last_hidden_state
        emo_feat = F.relu(self.audio_feature_map_emo(emo_hidden.detach()))
        emo_feat_bi, _ = self.emo_gru(emo_feat)

        # === FiLM Modulation ===
        modulated_feat = self.apply_film(cont_feat, emo_feat_bi)

        # === Embed Level + Person ===
        onehot_level = F.one_hot(level, num_classes=self.args.num_emotions).float()
        level_emb    = self.obj_vector_level(onehot_level)
        person_emb   = self.obj_vector_person(person)

        frame_num  = modulated_feat.size(1)
        level_emb  = level_emb.unsqueeze(1).repeat(1, frame_num, 1)
        person_emb = person_emb.unsqueeze(1).repeat(1, frame_num, 1)

        fused_input  = torch.cat([modulated_feat, level_emb, person_emb], dim=2)
        memory_input = self.emo_memory_proj(emo_feat_bi)

        tgt_mask    = self.biased_mask1[:, :frame_num, :frame_num]
        memory_mask = enc_dec_mask(self.device, frame_num, frame_num)

        # === Checkpoint each decoder layer ===
        out = fused_input
        for layer in self.decoder.layers:
            out = checkpoint(
                layer,
                out,
                memory_input,
                tgt_mask,
                memory_mask
            )
        if self.decoder.norm is not None:
            out = self.decoder.norm(out)

        bs_out    = out
        bs_output = self.linear(bs_out)

        # === Temporal Alignment ===
        if self.training and hasattr(self.args, 'target_len'):
            if bs_output.size(1) != self.args.target_len:
                bs_output = F.interpolate(
                    bs_output.transpose(1, 2),
                    size=self.args.target_len,
                    mode='linear',
                    align_corners=True
                ).transpose(1, 2)

        return bs_output

    def predict(self, audio, level, person):
        self.eval()
        with torch.no_grad():
            return self.forward(audio, level, person)
