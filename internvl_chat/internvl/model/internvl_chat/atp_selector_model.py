from enum import IntEnum
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
from typing import Optional

@dataclass
class ATPConfig:
    '''
    ATPConfig contains the parameters needed for the ATPSelectorModel (and its ATPEncoder).
    '''
    # ATPEncoder params
    n_layers: int = 2
    n_heads: int = 2
    d_model: int = 128
    d_model_ff: int = 128
    enc_dropout: float = 0.1
    use_text_query: bool = False  # at least one use_text_* needs to be true for ATP to be multimodal
    use_text_cands: bool = False  # ^ see above. (note: if both are false, ATP is vision-only)
    n_cands: int = 5  # only relevant when use_text_cands is set to true
    # ATPSelector params
    use_ste: bool = True  # controls type of selector during ATP training; see ATPSelectorModel.forward
    sel_dropout: float = 0.0
    d_input: int = 512  # size of the input vision-language embeddings (e.g. CLIP-ViT-B32 is size 512)

    @classmethod
    def from_args(cls, args):
        return cls(n_layers=args.n_layers,
                   n_heads=args.n_heads,
                   d_model=args.d_model,
                   d_model_ff=args.d_model_ff,
                   enc_dropout=args.enc_dropout,
                   use_text_query=args.use_text_query,
                   use_text_cands=args.use_text_cands,
                   n_cands=args.n_cands,
                   use_ste=args.use_ste,
                   sel_dropout=args.sel_dropout,
                   d_input=args.d_input)


class ModalityEmbeddingsID(IntEnum):
    TEXT_QUESTION = 0
    TEXT_EMBEDDING = 1
    TEXT_UNUSED = 2  # ignore
    VISUAL_EMBEDDING = 3
    VISUAL_UNUSED = 4  # ignore

class ModalityEmbeddings(nn.Module):
    """
    Provides embeddings that indicate type of modality; for use with multimodal inputs for ATP. See atp.py for usage.
    """
    def __init__(self,
                 d_model: int,
                 use_text_query: bool = False,
                 use_text_cands: bool = False,
                 n_cands: int = 5):
        """
        Details for each of these arguments are provided in ATPConfig.
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings=len(ModalityEmbeddingsID),
                                      embedding_dim=d_model)

        self.use_text_query = use_text_query
        self.use_text_cands = use_text_cands
        self.n_cands = n_cands if use_text_cands else 0
        self.n_text_feats = 1 if use_text_query else 0
        if use_text_cands:
            self.n_text_feats += n_cands

    def forward(self, x: torch.tensor):
        """
        x: torch.tensor of size (L, N, D)
        returns modality embeddings for x of size (L, *, D)
        """
        L, N, D = x.size()  # (sequence_length, batch_size, feature_dim)
        n_frames = L - self.n_text_feats

        # assemble the IDs for the modality encodings, language inputs then vision inputs
        class_ids = []
        if self.use_text_query:
            class_ids = [ModalityEmbeddingsID.TEXT_QUESTION, ]
        if self.use_text_cands:
            class_ids.extend([ModalityEmbeddingsID.TEXT_EMBEDDING, ] * self.n_cands)
        class_ids.extend([ModalityEmbeddingsID.VISUAL_EMBEDDING, ] * n_frames)

        class_ids = torch.tensor(
            class_ids,
            dtype=torch.long,
            device=x.device
        ).unsqueeze(-1)

        # return modality embeddings
        return self.embedding(class_ids)

class ATPEncoder(nn.Module):
    """
    The multimodal transformer encoder for the ATP model. For analysis purposes, the ATP encoder
    does not use any positional information (no positional encodings + transformer / self-attention)
    and is generally kept low-capacity. If the goal is raw accuracy (not analysis), you can relax these constraints.
    """
    def __init__(self, config: ATPConfig, **kwargs):
        """
        config: ATPConfig with parameters for the (transformer-based, atemporal) encoder for ATP.
        See ATPConfig documentation for details.
        """
        super().__init__()
        self.d_model = config.d_model
        self.dropout = nn.Dropout(p=config.enc_dropout)
        self.modality_encoding = ModalityEmbeddings(d_model=self.d_model,
                                                    use_text_query=config.use_text_query,
                                                    use_text_cands=config.use_text_cands,
                                                    n_cands=config.n_cands)

        atp_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model_ff,
            dropout=config.enc_dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(atp_encoder_layer, config.n_layers)

    def forward(self, x_inputs: torch.tensor):
        """
        x_inputs: torch.tensor of shape (L, N, D)
        """
        L, N, D = x_inputs.size()  # (sequence_length, batch_size, d_model)
        assert D == self.d_model, "inputs dimension mismatch"
        x_encoded = x_inputs * math.sqrt(self.d_model)
        x_encoded += self.modality_encoding(x_encoded)
        x_encoded = self.dropout(x_encoded)
        x_encoded = self.transformer_encoder(x_encoded)
        return x_encoded


class ATPSelectorModel(nn.Module):
    """
    The Atemporal Probe (ATP) selector model. Takes as input a sequence of image-language
    encoding and outputs a (discrete) selection over the input frames, to help analyze
    downstream discriminative video-language tasks.
    """
    def __init__(self, config: ATPConfig, **kwargs):
        """
        config: ATPConfig with parameters for initializing the ATPSelectorModel (and its encoder).
        See ATPConfig documentation for details.
        """
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(config.d_input, config.d_model)
        self.atp_encoder = ATPEncoder(config, **kwargs)
        self.dropout = nn.Dropout(p=config.sel_dropout)
        self.logits = nn.Linear(config.d_model, 1)

    def forward(self,
                x_vis_seq: torch.tensor,
                x_txt_query: Optional[torch.tensor] = None,
                x_txt_cands: Optional[torch.tensor] = None,
                **kwargs):
        """
        Performs the ATP selection operation on the input embeddings.
        Returns selected (unmodified) visual embeddings and selection mask.
        x_vis_seq: torch.tensor of shape (N, L, D_in) with visual embeddings of size D_in
        x_txt_query: torch.tensor of shape (N, D_in) with optional query text embeddings
        x_txt_cands: torch.tensor of shape (N, L_cands, D_in) with optional add'l text embeddings
        (optional) temperature: used when config.use_ste is set to False; set as keyword argument. Default = 1.0.
        """
        N, L, D = x_vis_seq.size()  # (batch_size, sequence_length, feature_dimension)
        x_vis_seq = x_vis_seq.permute(1, 0, 2)  # make (L, N, D); sequence first

        n_text_feats = self.atp_encoder.modality_encoding.n_text_feats

        # combine inputs into one multimodal sequence
        x_inputs = []
        if self.config.use_text_query:
            assert x_txt_query is not None, "missing x_txt_query."
            x_inputs.append(x_txt_query.unsqueeze(0))
        if self.config.use_text_cands:
            assert x_txt_cands is not None, "missing x_txt_cands."
            x_inputs.append(x_txt_cands.permute(1, 0, 2))
        x_inputs.append(x_vis_seq)
        x_inputs = torch.cat(x_inputs, dim=0)

        # embed the input sequence to the (smaller) model dimension (d_model) with modality encodings.
        x_encoded = self.embedding(self.dropout(x_inputs))
        x_atp_encoded = self.atp_encoder(x_encoded)[n_text_feats:]

        # obtain selection scores (logits)
        x_logits = self.logits(self.dropout(x_atp_encoded))

        # get selection mask over the visual inputs (frames)
        if self.training:
            # obtain differentiable selection mask during training.
            if self.config.use_ste:  # gumbel softmax straight-through estimator; hard selection
                selection_mask = F.gumbel_softmax(x_logits, dim=0, hard=True)
            else:  # softmax with temperature; soft selection
                selection_mask = F.softmax(x_logits / kwargs.get("temperature", 1.0), dim=0)
        else:
            # ATP always performs hard (discrete) selection during evaluation.
            selection_index_argmax = x_logits.max(dim=0, keepdim=True)[1]
            selection_mask = torch.zeros_like(x_logits, memory_format=torch.contiguous_format).scatter_(
                dim=0, index=selection_index_argmax, value=1.0)

        # use mask to perform selection
        selected_frames = (selection_mask * x_vis_seq).sum(dim=0)

        ret = [selected_frames, selection_mask]
        if not self.training:  # extra logging during validation
            ret.append(x_logits)
        return tuple(ret)