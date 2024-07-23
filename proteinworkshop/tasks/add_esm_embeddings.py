import os
import pdb
from typing import Dict, Set, Tuple, Union

import torch
from graphein.protein.tensor.data import ProteinBatch
from torch import Tensor
from torch_geometric import transforms as T
from torch_geometric.data import Batch


class AddESMEmbTransform(T.BaseTransform):
    """
    We attach pre-computed ESM embeddings to the batch of protein graphs.
    """

    def __init__(self, dataset: str, emb_dir: os.PathLike, model: str = "650M"):
        """Initialise the transform.

        :param emb_dir: Directory of the ESM embeddings
        :type emb_dir: os.PathLike
        """
        self.emb_dir = emb_dir
        self.dataset = dataset

        emb_path = os.path.join(self.emb_dir, f"{self.dataset}-{model}.pt")
        self.embs: Dict[str, Tuple[Tensor, Tensor]] = torch.load(
            emb_path, map_location="cpu"
        )
        pdb.set_trace()

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
        graph_emb, node_embs = self.embs[batch.id]
        batch.graph_esm_emb = graph_emb
        batch.node_esm_emb = node_embs
        return batch
