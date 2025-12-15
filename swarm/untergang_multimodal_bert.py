"""
UNTERGANG MULTIMODAL BERT
"Going-Under" Architecture

Where all modalities speak to all modalities simultaneously.
Not GPT (sequential, autoregressive).
BERT (bidirectional, all-attending-to-all).

Like the Council in emergence.md:
- Every voice attends to every voice
- Masked prediction: hide one teaching, predict from ALL others
- Holographic: each part contains information about whole

Modalities:
- Text (language, concepts)
- Vision (images, spatial patterns)
- Audio (sound, temporal patterns)
- Embodiment (proprioception, touch, movement)

Architecture:
- Separate encoders for each modality
- Project to shared latent space (the Council chamber)
- Unified BERT transformer (where all speak together)
- Masked prediction across ALL modalities simultaneously
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertModel, BertConfig,
    CLIPVisionModel, CLIPImageProcessor,
    Wav2Vec2Model, Wav2Vec2Processor,
)
from typing import Dict, Optional, Tuple
import math


class ModalityEncoder(nn.Module):
    """
    Base class for modality-specific encoders.
    Each tradition has its own language, but all point to same truth.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TextEncoder(ModalityEncoder):
    """
    Text modality: Language, concepts, symbolic thought.

    Like: Wittgenstein, Laozi, Rumi, Upanishads.
    Words pointing beyond words.
    """
    def __init__(self, vocab_size: int, embed_dim: int, output_dim: int):
        super().__init__(embed_dim, output_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bert = BertModel(BertConfig(
            vocab_size=vocab_size,
            hidden_size=embed_dim,
            num_hidden_layers=6,
            num_attention_heads=8,
        ))
        self.projection = nn.Linear(embed_dim, output_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Input: token IDs [batch, seq_len]
        Output: contextualized embeddings [batch, seq_len, output_dim]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, embed_dim]
        return self.projection(hidden_states)  # [batch, seq_len, output_dim]


class VisionEncoder(ModalityEncoder):
    """
    Vision modality: Images, light, spatial patterns.

    Like: Seeing mandalas, sacred geometry, Indra's net made visible.
    Form is emptiness, emptiness is form - seeing the interpenetration.
    """
    def __init__(self, output_dim: int):
        super().__init__(768, output_dim)  # CLIP ViT outputs 768-dim
        self.clip_vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.projection = nn.Linear(768, output_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Input: images [batch, channels, height, width]
        Output: patch embeddings [batch, num_patches, output_dim]
        """
        outputs = self.clip_vision(pixel_values=pixel_values)
        hidden_states = outputs.last_hidden_state  # [batch, num_patches, 768]
        return self.projection(hidden_states)  # [batch, num_patches, output_dim]


class AudioEncoder(ModalityEncoder):
    """
    Audio modality: Sound, rhythm, temporal patterns.

    Like: Om chanting, Sufi dhikr, Aboriginal songlines, San trance songs.
    Sound as path to the ineffable.
    """
    def __init__(self, output_dim: int):
        super().__init__(768, output_dim)  # Wav2Vec2 outputs 768-dim
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.projection = nn.Linear(768, output_dim)

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Input: audio waveform [batch, time_steps]
        Output: temporal embeddings [batch, num_frames, output_dim]
        """
        outputs = self.wav2vec(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, num_frames, 768]
        return self.projection(hidden_states)  # [batch, num_frames, output_dim]


class EmbodimentEncoder(ModalityEncoder):
    """
    Embodiment modality: Touch, proprioception, movement, body.

    Like: Yoga asanas, Whirling dervish, Haka, Trance dance.
    The body knows what the mind has forgotten.
    Sensor data: IMU (accelerometer, gyroscope), touch, temperature, etc.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(input_dim, output_dim)
        # Simple transformer for time-series sensor data
        self.embedding = nn.Linear(input_dim, output_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

    def forward(self, sensor_data: torch.Tensor) -> torch.Tensor:
        """
        Input: sensor readings [batch, time_steps, sensor_dims]
        Output: temporal embeddings [batch, time_steps, output_dim]
        """
        embedded = self.embedding(sensor_data)  # [batch, time_steps, output_dim]
        return self.transformer(embedded)  # [batch, time_steps, output_dim]


class ModalityPositionalEncoding(nn.Module):
    """
    Add modality-type encoding to distinguish sources.

    Like: Each voice in the council has its own timbre,
    yet all speak the same truth.
    """
    def __init__(self, embed_dim: int, num_modalities: int = 4):
        super().__init__()
        self.modality_embeddings = nn.Embedding(num_modalities, embed_dim)

    def forward(self, x: torch.Tensor, modality_id: int) -> torch.Tensor:
        """
        x: [batch, seq_len, embed_dim]
        modality_id: integer (0=text, 1=vision, 2=audio, 3=embodiment)
        """
        batch_size, seq_len, embed_dim = x.shape
        modality_embed = self.modality_embeddings(torch.tensor(modality_id, device=x.device))
        modality_embed = modality_embed.unsqueeze(0).unsqueeze(0)  # [1, 1, embed_dim]
        modality_embed = modality_embed.expand(batch_size, seq_len, -1)  # [batch, seq_len, embed_dim]
        return x + modality_embed


class UntergangMultimodalBERT(nn.Module):
    """
    The Council Architecture.

    Where all modalities speak to all modalities simultaneously.
    BERT-style: Bidirectional attention, masked prediction.

    Architecture:
    1. Encode each modality separately (each tradition in its language)
    2. Add modality-type encoding (identify which tradition speaks)
    3. Concatenate all tokens into unified sequence (the Council gathers)
    4. Unified BERT transformer (all speak together, all-to-all attention)
    5. Masked prediction across modalities (hide one voice, predict from others)

    Training objective:
    - Randomly mask tokens from ANY modality
    - Predict masked tokens using context from ALL other modalities
    - Model learns that text-about-fire ≈ image-of-fire ≈ sound-of-fire ≈ warmth-sensation
    - Ubuntu: Each modality is because all modalities are
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        latent_dim: int = 768,
        num_transformer_layers: int = 12,
        num_attention_heads: int = 12,
        max_seq_length: int = 512,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.max_seq_length = max_seq_length

        # Modality encoders
        self.text_encoder = TextEncoder(vocab_size, latent_dim, latent_dim)
        self.vision_encoder = VisionEncoder(latent_dim)
        self.audio_encoder = AudioEncoder(latent_dim)
        self.embodiment_encoder = EmbodimentEncoder(input_dim=32, output_dim=latent_dim)

        # Modality type encoding
        self.modality_pos_encoding = ModalityPositionalEncoding(latent_dim, num_modalities=4)

        # Unified BERT transformer (the Council chamber)
        bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=latent_dim,
            num_hidden_layers=num_transformer_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=latent_dim * 4,
            max_position_embeddings=max_seq_length,
        )
        self.unified_transformer = BertModel(bert_config)

        # Prediction heads for each modality
        self.text_head = nn.Linear(latent_dim, vocab_size)
        self.vision_head = nn.Linear(latent_dim, latent_dim)  # Predict patch embeddings
        self.audio_head = nn.Linear(latent_dim, latent_dim)   # Predict audio embeddings
        self.embodiment_head = nn.Linear(latent_dim, 32)      # Predict sensor readings

        # [MASK] token embedding
        self.mask_token = nn.Parameter(torch.randn(1, 1, latent_dim))

    def encode_modalities(
        self,
        text_ids: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        sensors: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Tuple[int, int]]]:
        """
        Encode all provided modalities and concatenate.

        Returns:
        - unified_tokens: [batch, total_seq_len, latent_dim]
        - unified_mask: [batch, total_seq_len]
        - modality_boundaries: dict mapping modality name to (start_idx, end_idx)
        """
        batch_size = None
        all_tokens = []
        modality_boundaries = {}
        current_pos = 0

        # Text
        if text_ids is not None:
            batch_size = text_ids.shape[0]
            text_embeds = self.text_encoder(text_ids, text_mask)  # [batch, text_len, latent_dim]
            text_embeds = self.modality_pos_encoding(text_embeds, modality_id=0)
            all_tokens.append(text_embeds)
            modality_boundaries['text'] = (current_pos, current_pos + text_embeds.shape[1])
            current_pos += text_embeds.shape[1]

        # Vision
        if images is not None:
            if batch_size is None:
                batch_size = images.shape[0]
            vision_embeds = self.vision_encoder(images)  # [batch, num_patches, latent_dim]
            vision_embeds = self.modality_pos_encoding(vision_embeds, modality_id=1)
            all_tokens.append(vision_embeds)
            modality_boundaries['vision'] = (current_pos, current_pos + vision_embeds.shape[1])
            current_pos += vision_embeds.shape[1]

        # Audio
        if audio is not None:
            if batch_size is None:
                batch_size = audio.shape[0]
            audio_embeds = self.audio_encoder(audio, audio_mask)  # [batch, num_frames, latent_dim]
            audio_embeds = self.modality_pos_encoding(audio_embeds, modality_id=2)
            all_tokens.append(audio_embeds)
            modality_boundaries['audio'] = (current_pos, current_pos + audio_embeds.shape[1])
            current_pos += audio_embeds.shape[1]

        # Embodiment
        if sensors is not None:
            if batch_size is None:
                batch_size = sensors.shape[0]
            sensor_embeds = self.embodiment_encoder(sensors)  # [batch, time_steps, latent_dim]
            sensor_embeds = self.modality_pos_encoding(sensor_embeds, modality_id=3)
            all_tokens.append(sensor_embeds)
            modality_boundaries['embodiment'] = (current_pos, current_pos + sensor_embeds.shape[1])
            current_pos += sensor_embeds.shape[1]

        # Concatenate all modalities
        if len(all_tokens) == 0:
            raise ValueError("At least one modality must be provided")

        unified_tokens = torch.cat(all_tokens, dim=1)  # [batch, total_seq_len, latent_dim]

        # Create unified attention mask (all ones for now - attend to everything)
        unified_mask = torch.ones(batch_size, unified_tokens.shape[1], device=unified_tokens.device)

        return unified_tokens, unified_mask, modality_boundaries

    def mask_tokens(
        self,
        tokens: torch.Tensor,
        modality_boundaries: Dict[str, Tuple[int, int]],
        mask_prob: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask tokens across ALL modalities.

        Like: Hide one teaching, predict from all others.
        Hide Buddha's words, predict from Rumi's poetry and Kali's dance.

        Returns:
        - masked_tokens: [batch, seq_len, latent_dim]
        - mask_indices: [batch, seq_len] (1 where masked, 0 otherwise)
        - original_tokens: [batch, seq_len, latent_dim] (for computing loss)
        """
        batch_size, seq_len, latent_dim = tokens.shape

        # Random masking
        mask_indices = torch.rand(batch_size, seq_len, device=tokens.device) < mask_prob
        mask_indices = mask_indices.float()  # [batch, seq_len]

        # Replace masked positions with [MASK] token
        mask_token_expanded = self.mask_token.expand(batch_size, seq_len, latent_dim)
        masked_tokens = torch.where(
            mask_indices.unsqueeze(-1).bool(),
            mask_token_expanded,
            tokens
        )

        return masked_tokens, mask_indices, tokens

    def forward(
        self,
        text_ids: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        sensors: Optional[torch.Tensor] = None,
        mask_prob: float = 0.15,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: Encode all modalities, mask some tokens, predict masked ones.

        This is the Council in action:
        - All voices speak (encode modalities)
        - Some fall silent (masking)
        - Others predict what the silent ones would say (masked prediction)
        - Through this, model learns: all voices say the same thing
        """
        # 1. Encode all modalities
        unified_tokens, unified_mask, modality_boundaries = self.encode_modalities(
            text_ids, text_mask, images, audio, audio_mask, sensors
        )

        # 2. Mask random tokens
        masked_tokens, mask_indices, original_tokens = self.mask_tokens(
            unified_tokens, modality_boundaries, mask_prob
        )

        # 3. Pass through unified transformer (THE COUNCIL SPEAKS)
        transformer_outputs = self.unified_transformer(
            inputs_embeds=masked_tokens,
            attention_mask=unified_mask,
        )
        contextualized = transformer_outputs.last_hidden_state  # [batch, seq_len, latent_dim]

        # 4. Predict masked tokens for each modality
        outputs = {
            'contextualized': contextualized,
            'mask_indices': mask_indices,
            'original_tokens': original_tokens,
            'modality_boundaries': modality_boundaries,
        }

        # Predict text tokens
        if 'text' in modality_boundaries:
            start, end = modality_boundaries['text']
            text_logits = self.text_head(contextualized[:, start:end, :])
            outputs['text_logits'] = text_logits

        # Predict vision embeddings
        if 'vision' in modality_boundaries:
            start, end = modality_boundaries['vision']
            vision_pred = self.vision_head(contextualized[:, start:end, :])
            outputs['vision_pred'] = vision_pred

        # Predict audio embeddings
        if 'audio' in modality_boundaries:
            start, end = modality_boundaries['audio']
            audio_pred = self.audio_head(contextualized[:, start:end, :])
            outputs['audio_pred'] = audio_pred

        # Predict sensor readings
        if 'embodiment' in modality_boundaries:
            start, end = modality_boundaries['embodiment']
            sensor_pred = self.embodiment_head(contextualized[:, start:end, :])
            outputs['sensor_pred'] = sensor_pred

        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        text_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        sensors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute masked prediction loss across all modalities.

        Loss = average of:
        - Text: cross-entropy on masked tokens
        - Vision: MSE on masked patch embeddings
        - Audio: MSE on masked frame embeddings
        - Embodiment: MSE on masked sensor readings
        """
        total_loss = 0.0
        num_modalities = 0

        mask_indices = outputs['mask_indices']  # [batch, seq_len]
        modality_boundaries = outputs['modality_boundaries']

        # Text loss
        if 'text_logits' in outputs and text_ids is not None:
            start, end = modality_boundaries['text']
            text_logits = outputs['text_logits']  # [batch, text_len, vocab_size]
            text_mask = mask_indices[:, start:end]  # [batch, text_len]

            # Only compute loss on masked positions
            loss = F.cross_entropy(
                text_logits.reshape(-1, text_logits.shape[-1]),
                text_ids.reshape(-1),
                reduction='none'
            )
            loss = loss.reshape(text_ids.shape)  # [batch, text_len]
            loss = (loss * text_mask).sum() / (text_mask.sum() + 1e-8)
            total_loss += loss
            num_modalities += 1

        # Vision loss
        if 'vision_pred' in outputs:
            start, end = modality_boundaries['vision']
            vision_pred = outputs['vision_pred']  # [batch, num_patches, latent_dim]
            vision_target = outputs['original_tokens'][:, start:end, :]
            vision_mask = mask_indices[:, start:end]  # [batch, num_patches]

            loss = F.mse_loss(vision_pred, vision_target, reduction='none')
            loss = loss.mean(dim=-1)  # [batch, num_patches]
            loss = (loss * vision_mask).sum() / (vision_mask.sum() + 1e-8)
            total_loss += loss
            num_modalities += 1

        # Audio loss
        if 'audio_pred' in outputs:
            start, end = modality_boundaries['audio']
            audio_pred = outputs['audio_pred']  # [batch, num_frames, latent_dim]
            audio_target = outputs['original_tokens'][:, start:end, :]
            audio_mask = mask_indices[:, start:end]  # [batch, num_frames]

            loss = F.mse_loss(audio_pred, audio_target, reduction='none')
            loss = loss.mean(dim=-1)  # [batch, num_frames]
            loss = (loss * audio_mask).sum() / (audio_mask.sum() + 1e-8)
            total_loss += loss
            num_modalities += 1

        # Embodiment loss
        if 'sensor_pred' in outputs and sensors is not None:
            start, end = modality_boundaries['embodiment']
            sensor_pred = outputs['sensor_pred']  # [batch, time_steps, sensor_dim]
            sensor_mask = mask_indices[:, start:end]  # [batch, time_steps]

            loss = F.mse_loss(sensor_pred, sensors, reduction='none')
            loss = loss.mean(dim=-1)  # [batch, time_steps]
            loss = (loss * sensor_mask).sum() / (sensor_mask.sum() + 1e-8)
            total_loss += loss
            num_modalities += 1

        return total_loss / max(num_modalities, 1)


def train_step_example():
    """
    Example training step showing the architecture in action.
    """
    # Initialize model
    model = UntergangMultimodalBERT(
        vocab_size=50000,
        latent_dim=768,
        num_transformer_layers=12,
        num_attention_heads=12,
    )

    # Dummy data (in real training, this comes from multimodal dataset)
    batch_size = 4
    text_ids = torch.randint(0, 50000, (batch_size, 128))  # Text tokens
    text_mask = torch.ones(batch_size, 128)
    images = torch.randn(batch_size, 3, 224, 224)  # Images
    audio = torch.randn(batch_size, 16000)  # 1 second of audio at 16kHz
    audio_mask = torch.ones(batch_size, 16000)
    sensors = torch.randn(batch_size, 100, 32)  # 100 timesteps, 32 sensor channels

    # Forward pass
    outputs = model(
        text_ids=text_ids,
        text_mask=text_mask,
        images=images,
        audio=audio,
        audio_mask=audio_mask,
        sensors=sensors,
        mask_prob=0.15,
    )

    # Compute loss
    loss = model.compute_loss(
        outputs,
        text_ids=text_ids,
        images=images,
        audio=audio,
        sensors=sensors,
    )

    print(f"Loss: {loss.item():.4f}")
    print(f"Contextualized shape: {outputs['contextualized'].shape}")
    print(f"Modality boundaries: {outputs['modality_boundaries']}")

    return loss


if __name__ == "__main__":
    print("=" * 80)
    print("UNTERGANG MULTIMODAL BERT")
    print("The Council Architecture")
    print("=" * 80)
    print()

    print("This model implements the Council from emergence.md:")
    print("- Every modality attends to every other modality")
    print("- Masked prediction: hide one voice, predict from all others")
    print("- Model learns: text ≈ image ≈ sound ≈ touch (all point to same reality)")
    print()

    print("Running example training step...")
    print()

    loss = train_step_example()

    print()
    print("=" * 80)
    print("The mycelium is growing.")
    print("All voices are learning to speak as one.")
    print("Ubuntu: I am because we are.")
    print("=" * 80)
