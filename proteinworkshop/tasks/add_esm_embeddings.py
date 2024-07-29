import os
import pdb
from typing import Dict, Set, Tuple, Union

import torch
from graphein.protein.tensor.data import ProteinBatch
from torch import Tensor
from torch_geometric import transforms as T
from torch_geometric.data import Batch

output_dim: Dict[str, int] = {
    "ESM-1b": 1280,
    "ESM-1v": 1280,
    "ESM-2-8M": 320,
    "ESM-2-35M": 480,
    "ESM-2-150M": 640,
    "ESM-2-650M": 1280,
    "ESM-2-3B": 2560,
    "ESM-2-15B": 5120,
}


class AddESMEmbTransform(T.BaseTransform):
    """
    We attach pre-computed ESM embeddings to the batch of protein graphs.
    """

    def __init__(self, dataset: str, emb_dir: os.PathLike, model: str = "ESM-2-650M"):
        """Initialise the transform.

        :param emb_dir: Directory of the ESM embeddings
        :type emb_dir: os.PathLike
        """
        self.emb_dir = emb_dir
        self.dataset = dataset
        self.emb_dim = output_dim[model]

        emb_path = os.path.join(self.emb_dir, f"{self.dataset}-{model}.pt")
        self.embs: Dict[str, Tuple[Tensor, Tensor]] = torch.load(
            emb_path, map_location="cpu"
        )

    @property
    def required_batch_attributes(self) -> Set[str]:
        """
        Returns the set of attributes that this transform requires to be
        present on the batch object for correct operation.

        :return: Set of required attributes
        :rtype: Set[str]
        """
        return {
            "id",
            "residues",
            "batch",
        }

    def __call__(self, batch: Union[ProteinBatch, Batch]) -> Union[Batch, ProteinBatch]:
        if isinstance(batch.id, list):
            node_esm_emb = torch.zeros((batch.num_nodes, self.emb_dim))
            for i, id in enumerate(batch.id):
                graph_emb, node_embs = self.embs[id]
                node_esm_emb[batch.batch == i] = node_embs
        else:
            graph_emb, node_esm_emb = self.embs[batch.id]

        batch.node_esm_emb = node_esm_emb
        batch.graph_esm_emb = graph_emb
        return batch
