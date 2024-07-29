import os
import pdb
from typing import Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from graphein.protein.resi_atoms import RESI_THREE_TO_1
from graphein.protein.tensor.data import ProteinBatch
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_add

from proteinworkshop.models.utils import (
    get_aggregation,
    padded_to_variadic,
    variadic_to_padded,
)
from proteinworkshop.types import EncoderOutput

from .esm_embeddings import EvolutionaryScaleModeling
from .gear_net import GearNet

# from .gvp import GVPGNNModel


class MHAPyTorchScaledDotProduct(nn.Module):
    def __init__(
        self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False
    ):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x, mask):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0.0 if not self.training else self.dropout
        context_vec = nn.functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=mask,
            dropout_p=use_dropout,
            is_causal=False,
        )

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = (
            context_vec.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_tokens, self.d_out)
        )

        context_vec = self.proj(context_vec)

        return context_vec


class SelfAttentionBlock(nn.Module):
    """
    Multi-head self-attention block from
    `Attention Is All You Need`_.

    .. _Attention Is All You Need:
        https://arxiv.org/pdf/1706.03762.pdf

    Parameters:
        hidden_dim (int): hidden dimension
        num_heads (int): number of attention heads
        dropout (float, optional): dropout ratio of attention maps
    """

    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super(SelfAttentionBlock, self).__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_dim, num_heads)
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        # self.attn = MHAPyTorchScaledDotProduct(
        #     hidden_dim, hidden_dim, num_heads, hidden_dim, dropout=dropout
        # )

    def forward(self, src, tgt, mask):
        """
        Perform self attention over the input.

        Parameters:
            input (Tensor): input representations of shape `(..., length, dim)`
            mask (Tensor): bool mask of shape `(..., length)`
        """
        query = self.query(src).transpose(0, 1)
        key = self.key(tgt).transpose(0, 1)
        value = self.value(tgt).transpose(0, 1)

        if mask is not None:
            mask = (~mask.bool()).squeeze(-1)

        output = self.attn(query, key, value, key_padding_mask=mask)[0].transpose(0, 1)
        # output = self.attn(input, mask)

        return output


class SequenceStructInteractor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        sequence_model: str,
        structure_model: str,
        num_relation: int,
        num_struct_layers: int = 6,
        n_heads: int = 8,
        cross_dim: int = 512,
        activation: str = "relu",
        norm: str = "layer",
        aggr: str = "sum",
        pool: str = "sum",
        residual: bool = True,
        strategy: str = "interspersed",
    ):
        super().__init__()
        # self.sequence_model_name = sequence_model
        self.sequence_model = EvolutionaryScaleModeling(
            path=os.environ.get("DATA_PATH"),
            model=sequence_model,
            finetune=False,
            mlp_post_embed=False,
        )
        seq_dim = EvolutionaryScaleModeling.output_dim[sequence_model]

        self.structure_model_name = structure_model
        if structure_model == "GVP":
            # self.structure_model = GVPGNNModel(
            #     num_layers=num_struct_layers, s_dim=cross_dim
            # )
            pass
        elif structure_model == "GearNet":
            self.structure_model = GearNet(
                input_dim=input_dim,
                num_relation=num_relation,
                num_layers=num_struct_layers,
                emb_dim=cross_dim,
                short_cut=True,
                concat_hidden=False,
                batch_norm=True,
                num_angle_bin=7,
            )
        else:
            raise ValueError("Invalid structure model")

        self.interactors = nn.ModuleList(
            # SelfAttentionBlock(cross_dim, num_heads=n_heads, dropout=0.1).to(
            # torch.device("cuda")
            # )
            SelfAttentionBlock(cross_dim, num_heads=n_heads, dropout=0.1)
            for _ in range(num_struct_layers)
        )
        self.num_struct_layers = num_struct_layers

        self.readout = get_aggregation(pool)
        # self.fuser = nn.Linear(cross_dim, cross_dim).to(torch.device("cuda"))
        self.fuser = nn.Linear(cross_dim, cross_dim)

        # self.sequence_model.model = self.sequence_model.model.to(torch.device("cuda"))
        # self.structure_model = self.structure_model.to(torch.device("cuda"))

        # self.down_projector_seq = nn.Linear(seq_dim, cross_dim).to(torch.device("cuda"))
        self.down_projector_seq = nn.Linear(seq_dim, cross_dim)
        # self.down_projector_struct = nn.Linear(
        #     cross_dim * num_struct_layers, cross_dim
        # ).to(torch.device("cuda"))
        self.down_projector_struct = nn.Linear(cross_dim * num_struct_layers, cross_dim)

        self.pe_encoder = nn.LazyLinear(seq_dim)

        self.strategy = strategy
        if self.strategy not in ("interspersed", "last", "all"):
            raise ValueError("Invalid strategy")

    @property
    def required_batch_attributes(self) -> Set[str]:
        return {
            "x",
            "pos",
            "edge_index",
            "coords",
            "id",
            "residues",
            "batch",
            "edge_type",
            "edge_attr",
            "num_nodes",
            # "graph_esm_emb",
            "node_esm_emb",
        }

    def tokenize(self, batch: Union[Batch, ProteinBatch]) -> torch.Tensor:
        device = batch.coords.device

        seqs = [
            "".join([self.sequence_model.residue_map[s] for s in seq])
            for seq in batch.residues
        ]
        seqs = ["".join(seq) for seq in seqs]
        data = list(tuple(zip(batch.id, seqs)))

        _, _, batch_tokens = self.sequence_model.batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        return batch_tokens

    def get_plm_embs(self, batch_tokens: torch.Tensor) -> torch.Tensor:
        repr_layers = range(self.sequence_model.repr_layer)
        return [
            [
                *self.sequence_model.model(batch_tokens, repr_layers=repr_layers)[
                    "representations"
                ].values()
            ][idx]
            for idx in range(self.num_struct_layers)
        ]

    def struct_setup(
        self, batch: Union[Batch, ProteinBatch]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.structure_model_name == "GVP":
            vectors = (
                batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]
            )  # [n_edges, 3]
            lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
            h_V = self.structure_model.emb_in(batch.x)
            h_E = (
                self.structure_model.radial_embedding(lengths),
                torch.nan_to_num(torch.div(vectors, lengths)).unsqueeze_(-2),
            )

            h_V = self.structure_model.W_v(h_V)
            h_E = self.structure_model.W_e(h_E)

            return h_V, h_E
        elif self.structure_model_name == "GearNet":
            layer_input, edge_input, line_graph, batch = (
                self.structure_model.setup_forward(batch)
            )
            return layer_input, edge_input, line_graph, batch

    def forward(
        self,
        batch: Union[Batch, ProteinBatch],
    ) -> EncoderOutput:
        # batch = batch.to(torch.device("cuda"))

        # node_seq = self.down_projector_seq(self.sequence_model(batch)["node_embedding"])
        # node_emb = self.structure_model(batch)["node_embedding"]

        # node_emb = torch.mean(torch.stack([node_seq, node_struct]), dim=0)
        # node_emb = self.fuser(node_emb)

        # return EncoderOutput(
        #     {
        #         "node_embedding": node_emb,
        #         "graph_embedding": self.readout(node_emb, batch.batch),
        #     }
        # )

        # tokens = self.tokenize(batch)

        # x_seq = self.sequence_model.model.embed_tokens(tokens).transpose(0, 1)

        # padding_idx = self.sequence_model.model.padding_idx
        # padding_mask = tokens.eq(padding_idx)

        if self.structure_model_name == "GVP":
            x_struct_V, x_struct_E = self.struct_setup(batch)
        elif self.structure_model_name == "GearNet":
            x_struct_V, x_struct_E, line_graph, batch_modified = self.struct_setup(
                batch
            )
            batch = batch_modified

        # if self.strategy == "interspersed":
        #     layer_indices = list(
        #         i
        #         for i in range(self.sequence_model.repr_layer)
        #         if i % self.num_struct_layers == 0
        #     )
        # elif self.strategy == "last":
        #     layer_indices = [self.sequence_model.repr_layer - 1]
        # elif self.strategy == "all":
        #     layer_indices = list(range(self.sequence_model.repr_layer))

        # with torch.no_grad():
        #     x_seqs = list(
        #         self.sequence_model.model(tokens, repr_layers=layer_indices)[
        #             "representations"
        #         ].values()
        #     )

        # num_residues = torch.tensor([len(x) for x in batch.residues]).to(
        #     self.structure_model.device
        # )
        if batch.x.shape[0] == 39:
            # pe, _ = variadic_to_padded(batch.x[:, :16], num_residues)
            pe = batch.x[:, :16]
        else:
            # pe, _ = variadic_to_padded(batch.x, num_residues)
            pe = batch.x

        # if self.strategy == "all":
        #     x_seq = torch.mean(torch.stack(x_seqs), dim=0)
        #     x_seq = x_seq[:, 1 : x_seq.shape[1] - 1, :]
        #     x_seq = self.pe_encoder(torch.cat([x_seq, pe], dim=-1))
        #     x_seq = self.down_projector_seq(x_seq)
        # elif self.strategy == "last":
        #     x_seq = x_seqs[0]
        #     x_seq = x_seq[:, 1 : x_seq.shape[1] - 1, :]
        #     x_seq = self.pe_encoder(torch.cat([x_seq, pe], dim=-1))
        #     x_seq = self.down_projector_seq(x_seq)
        # pdb.set_trace()

        x_seq = batch.node_esm_emb

        with torch.no_grad():
            from_plm = self.sequence_model(batch)["node_embedding"]

        # if x_seq.shape[0] != pe.shape[0]:
        #     pdb.set_trace()
        #     x_seq = x_seq[:-1, :]

        # x_seq = self.pe_encoder(torch.cat([x_seq, pe], dim=-1))
        x_seq = self.pe_encoder(torch.cat([from_plm, pe], dim=-1))
        x_seq = self.down_projector_seq(x_seq)

        for i in range(self.num_struct_layers):
            # if self.strategy == "interspersed":
            # x_seq = x_seqs[i]
            # x_seq = batch.node_esm_emb[1:, :]
            # x_seq = torch.cat([self.embs[x][1] for x in batch.id_with_chain]).to(
            #     batch.x.device
            # )[1:, :]
            # x_seq = torch.cat([self.embs[x][1] for x in batch.id]).to(
            #     batch.x.device
            # )[1:, :]

            if self.structure_model_name == "GVP":
                x_struct_V, x_struct_E = self.structure_model.layers[i](
                    x_struct_V, batch.edge_index, x_struct_E
                )
            elif self.structure_model_name == "GearNet":
                x_struct_V, x_struct_E = self.structure_model.one_layer_computation(
                    batch, i, x_struct_V, line_graph, x_struct_E
                )

            # x_struct_V, mask_struct_ = variadic_to_padded(x_struct_V, num_residues)

            # cat = torch.cat([x_seq, x_struct_V], dim=1)
            # mask_seq = padding_mask[:, 1 : padding_mask.shape[1] - 1]
            # mask = torch.cat([mask_seq, mask_struct_], dim=1)
            # attn_output = self.interactors[layer_idx_struct](cat, mask)

            # x_struct_V = self.interactors[i](x_seq, x_struct_V, mask_struct_)
            num_residues = torch.tensor([len(x) for x in batch.residues]).to(
                self.structure_model.device
            )
            # pdb.set_trace()
            x_seq_var = variadic_to_padded(x_seq, num_residues)
            x_struct_var = variadic_to_padded(x_seq, num_residues)
            x_struct_V = self.interactors[i](
                x_seq_var[0], x_struct_var[0], x_struct_var[1]
            )

            # x_seq, x_struct_V = torch.split(attn_output, n, dim=1)
            # x_struct_V = torch.mean(torch.stack([x_seq, x_struct_V]), dim=0)

            x_struct_V = padded_to_variadic(x_struct_V, num_residues)

        # for layer_idx_seq, layer_seq in enumerate(self.sequence_model.model.layers):
        #     x_seq, _ = layer_seq(
        #         x_seq,
        #         self_attn_padding_mask=padding_mask,
        #         need_head_weights=False,
        #     )  # N x B x F

        #     x_seq = x_seq.transpose(0, 1)  # B x N x F

        #     if layer_idx_seq % self.num_struct_layers == 0:
        #         x_seq_down = x_seq[
        #             :, 1 : x_seq.shape[1] - 1, :
        #         ]  # remove bos and eos toks
        #         x_seq_down = x_seq_down.to(self.structure_model.device)
        #         x_seq_down = self.down_projector_seq(x_seq_down)

        #         if self.structure_model_name == "GVP":
        #             x_struct_V, x_struct_E = self.structure_model.layers[
        #                 layer_idx_struct
        #             ](
        #                 x_struct_V, batch.edge_index, x_struct_E
        #             )  # x_struct_V (sum_{b \in B} N_b x B) x F
        #         elif self.structure_model_name == "GearNet":
        #             x_struct_V, x_struct_E = self.structure_model.one_layer_computation(
        #                 batch, layer_idx_struct, x_struct_V, line_graph, x_struct_E
        #             )

        #         n = x_seq_down.shape[1]
        # num_residues = torch.tensor([len(x) for x in batch.residues]).to(
        #     self.structure_model.device
        # )
        #         x_struct_V, mask_struct_ = variadic_to_padded(
        #             x_struct_V, num_residues
        #         )  # B x N-2 x F, - start and filler toks

        #         cat = torch.cat([x_seq_down, x_struct_V], dim=1)  # B x 2N x F
        #         mask_seq = padding_mask[:, 1 : padding_mask.shape[1] - 1]
        #         mask = torch.cat([mask_seq, mask_struct_], dim=1)  # B x 2N
        #         attn_output = self.interactors[layer_idx_struct](cat, mask)
        #         x_seq_down, x_struct_V = torch.split(attn_output, n, dim=1)

        #         x_struct_V = padded_to_variadic(x_struct_V, num_residues)

        #         layer_idx_struct += 1

        #     x_seq = x_seq.transpose(0, 1)

        if self.structure_model_name == "GVP":
            x_struct = self.structure_model.W_out(x_struct_V)
        elif self.structure_model_name == "GearNet":
            x_struct = x_struct_V

        # x_struct, mask = variadic_to_padded(x_struct, num_residues)
        # x = torch.mean(torch.stack([x_seq, x_struct]), dim=0)

        # x = padded_to_variadic(x, num_residues)
        x = x_struct

        return EncoderOutput(
            {
                "node_embedding": x,
                "graph_embedding": self.readout(x, batch.batch),
            }
        )
