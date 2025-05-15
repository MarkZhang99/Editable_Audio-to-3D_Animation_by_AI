import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from utils import init_biased_mask, enc_dec_mask

class EmoTalkUpgrade(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # === Content Feature Extractor ===
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        )
        self.extractor = Wav2Vec2Model.from_pretrained(
            "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        )
        self.extractor_proj = nn.Linear(1024, args.transformer_dim)

        # === Emotion Feature Extractor ===
        self.emo_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "r-f/wav2vec-english-speech-emotion-recognition"
        )
        self.emo_extractor = Wav2Vec2Model.from_pretrained(
            "r-f/wav2vec-english-speech-emotion-recognition"
        )
        self.audio_feature_map_emo = nn.Linear(1024, args.transformer_dim)
        self.emo_gru = nn.GRU(
            input_size=args.transformer_dim,
            hidden_size=args.emo_gru_hidden,
            num_layers=args.emo_gru_layers,
            bidirectional=True,
            batch_first=True
        )

        # === FiLM Parameters ===
        self.film_gamma = nn.Linear(args.emo_gru_hidden * 2, args.transformer_dim)
        self.film_beta  = nn.Linear(args.emo_gru_hidden * 2, args.transformer_dim)

        # === Level + Person Embedding ===
        self.obj_vector_level  = nn.Embedding(args.num_emotions, 32)
        self.obj_vector_person = nn.Embedding(args.num_person,   32)

        # === Optional Prosody Integration ===
        if getattr(args, "use_prosody", False):
            self.prosody_proj = nn.Linear(2, args.transformer_dim)
            self.final_proj   = nn.Linear(args.transformer_dim * 2, args.transformer_dim)
        else:
            self.prosody_proj = None
            self.final_proj   = None

        # === Transformer Decoder + Dropout ===
        decoder_dim = args.transformer_dim + 64  # modulated_feat + level + person
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=args.transformer_heads,
            dim_feedforward=decoder_dim * 4,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.transformer_layers)
        self.dropout = nn.Dropout(p=args.dropout)
        self.bs_map_r = nn.Linear(decoder_dim, args.bs_dim)

        # === Emotion Classification Head ===
        self.emotion_cls = nn.Linear(args.emo_gru_hidden * 2, args.num_emotions)

        # === Masks & Memory Projection ===
        self.biased_mask1 = init_biased_mask(
            n_head=args.transformer_heads,
            max_seq_len=args.max_seq_len,
            period=args.period
        )
        self.emo_memory_proj = nn.Linear(args.emo_gru_hidden * 2, decoder_dim)

    def apply_film(self, content_feat, emotion_feat):
        gamma = self.film_gamma(emotion_feat)
        beta  = self.film_beta(emotion_feat)
        return gamma * content_feat + beta

    def forward(self, audio, level, person, prosody=None):
        # --- Audio preprocess ---
        if audio.dim() == 4:
            audio = audio.squeeze(1).squeeze(1)
        elif audio.dim() == 3:
            audio = audio.squeeze(1)
        elif audio.dim() != 2:
            raise ValueError(f"Unsupported audio shape: {audio.shape}")

        # === Content Features ===
        ivals = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")['input_values']
        ivals = ivals.to(audio.device).squeeze(1)
        crepr = self.extractor(ivals).last_hidden_state
        cfeat = self.extractor_proj(crepr)  # [B, T, D]

        # === Optional Prosody ===
        if prosody is not None and self.prosody_proj:
            pf = self.prosody_proj(prosody.to(audio.device))
            cfeat = self.final_proj(torch.cat([cfeat, pf], dim=-1))

        # === Emotion Features ===
        evals = self.emo_feature_extractor(audio, sampling_rate=16000, return_tensors="pt")['input_values']
        evals = evals.to(audio.device).squeeze(1)
        erepr = self.emo_extractor(evals).last_hidden_state
        efeat = F.relu(self.audio_feature_map_emo(erepr))
        efeat_bi, _ = self.emo_gru(efeat)  # [B, T, 2H]

        # === FiLM Modulation ===
        modfeat = self.apply_film(cfeat, efeat_bi)

        # === Embeddings ===
        lvl_emb = self.obj_vector_level(level.to(audio.device))
        prs_emb = self.obj_vector_person(person.to(audio.device))
        T = modfeat.size(1)
        lvl_emb = lvl_emb.unsqueeze(1).repeat(1, T, 1)
        prs_emb = prs_emb.unsqueeze(1).repeat(1, T, 1)

        # === Decoder Input ===
        x = torch.cat([modfeat, lvl_emb, prs_emb], dim=-1)
        x = self.dropout(x)
        mem = self.emo_memory_proj(efeat_bi)

        # === Masks ===
        tgt = self.biased_mask1[:, :T, :T].to(audio.device)
        mem_mask = enc_dec_mask(audio.device, T, T)

        # === Decode & Blendshape Output ===
        dout = self.decoder(x, mem, tgt_mask=tgt, memory_mask=mem_mask)
        bs = self.bs_map_r(dout)

        # === Emotion Classification ===
        pooled = efeat_bi.mean(dim=1)
        cls_logits = self.emotion_cls(pooled)

        return bs, cls_logits

    def predict(self, audio, level, person, prosody=None):
        self.eval()
        with torch.no_grad():
            bs, _ = self.forward(audio, level, person, prosody)
        return bs

# Notes:
# - args must include: dropout (float), use_prosody (bool)
# - forward prosody arg should be [B, T, 2] if use_prosody=True
