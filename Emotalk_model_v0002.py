import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from utils import init_biased_mask, enc_dec_mask

class EmoTalk(nn.Module):
    def __init__(self, args):
        super(EmoTalk, self).__init__()
        self.args = args
        self.device = args.device

        # === Content Feature Extractor ===
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        self.extractor = Wav2Vec2Model.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english").to(self.device)
        self.extractor_proj = nn.Linear(1024, args.transformer_dim).to(self.device)

        # === Emotion Feature Extractor ===
        self.emo_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")
        self.emo_extractor = Wav2Vec2Model.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition").to(self.device)
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
        self.film_beta = nn.Linear(args.emo_gru_hidden * 2, args.transformer_dim).to(self.device)

        # === Level + Person Embedding ===
        self.obj_vector_level = nn.Linear(args.num_emotions, 32).to(self.device)
        self.obj_vector_person = nn.Embedding(args.num_person, 32).to(self.device)

        # === Transformer Decoder ===
        decoder_input_dim = args.transformer_dim + 64  # modulated_feat + level + person
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_input_dim,
            nhead=args.transformer_heads,
            dim_feedforward=decoder_input_dim * 4,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.transformer_layers).to(self.device)
        self.linear = nn.Linear(decoder_input_dim, args.bs_dim).to(self.device)

        # === Attention Mask ===
        self.biased_mask1 = init_biased_mask(
            n_head=args.transformer_heads,
            max_seq_len=args.max_seq_len,
            period=args.period
        ).to(self.device)

        # === Memory Projection for Decoder Matching ===
        self.emo_memory_proj = nn.Linear(args.emo_gru_hidden * 2, decoder_input_dim).to(self.device)

    def apply_film(self, content_feat, emotion_feat):
        gamma = self.film_gamma(emotion_feat)
        beta = self.film_beta(emotion_feat)
        return gamma * content_feat + beta
        

    def forward(self, audio, level, person):
        print(f"\U0001f440 Input audio shape (preprocess): {audio.shape}")
        if audio.dim() == 4:
            audio = audio.squeeze(1).squeeze(1)
        elif audio.dim() == 3:
            audio = audio.squeeze(1)
        elif audio.dim() == 2:
            pass
        else:
            raise ValueError(f"Unsupported audio shape: {audio.shape}")

        # === Content Features ===
        input_values = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")['input_values'].to(self.device)
        input_values = input_values.squeeze(1)
        cont_feat = self.extractor(input_values).last_hidden_state
        cont_feat = self.extractor_proj(cont_feat)

        # === Emotion Features ===
        emo_input = self.emo_feature_extractor(audio, sampling_rate=16000, return_tensors='pt')['input_values'].to(self.device)
        emo_input = emo_input.squeeze(1)
        emo_feat = self.emo_extractor(emo_input).last_hidden_state
        emo_feat = F.relu(self.audio_feature_map_emo(emo_feat))
        emo_feat_bi, _ = self.emo_gru(emo_feat)

        # === FiLM Modulation ===
        modulated_feat = self.apply_film(cont_feat, emo_feat_bi)

        # === Embedding Level + Person ===
        level = level.to(self.device)
        person = person.to(self.device)
        onehot_level = F.one_hot(level, num_classes=self.args.num_emotions).float()
        level_emb = self.obj_vector_level(onehot_level)
        person_emb = self.obj_vector_person(person)

        if not ((level >= 0).all() and (level < self.args.num_emotions).all()):
            print("🚨 非法情绪标签 level:", level)
            raise ValueError("level 超出 num_emotions 范围")


        frame_num = modulated_feat.size(1)
        level_emb = level_emb.unsqueeze(1).repeat(1, frame_num, 1)
        person_emb = person_emb.unsqueeze(1).repeat(1, frame_num, 1)

        fused_input = torch.cat([modulated_feat, level_emb, person_emb], dim=2)
        memory_input = self.emo_memory_proj(emo_feat_bi)

        print("\U0001f3af fused_input:", fused_input.shape)
        print("\U0001f3af memory_input:", memory_input.shape)

        tgt_mask = self.biased_mask1[:, :frame_num, :frame_num].clone().detach().to(self.device)
        memory_mask = enc_dec_mask(self.device, frame_num, frame_num)

        bs_out = self.decoder(fused_input, memory_input, tgt_mask=tgt_mask, memory_mask=memory_mask)
        bs_output = self.linear(bs_out)
        
        # === Temporal Alignment ===
        # === Temporal Alignment（仅训练时补帧）===
        if self.training:
            if hasattr(self.args, 'target_len') and self.args.target_len > 0:
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
    



