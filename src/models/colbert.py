from typing import Dict
import torch
from transformers import PreTrainedModel, PretrainedConfig
from transformers import AutoModel


class ColBERTConfig(PretrainedConfig):
    model_type = "ColBERT"
    bert_model: str
    compression_dim: int = 768
    dropout: float = 0.0
    return_vecs: bool = False
    trainable: bool = False


class ColBERT(PreTrainedModel):
    """
    ColBERT model from: https://arxiv.org/pdf/2004.12832.pdf
    """
    
    config_class = ColBERTConfig
    base_model_prefix = "bert_model"
    is_teacher_model = False

    @staticmethod
    def from_config(config):
        # colbert_compression_dim: 768
        cfg = ColBERTConfig()
        cfg.bert_model = config["bert_pretrained_model"]
        cfg.compression_dim = config["colbert_compression_dim"]
        cfg.return_vecs = config.get("in_batch_negatives", False)
        cfg.trainable = config["bert_trainable"]
        return ColBERT(cfg)
    
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.return_vecs = cfg.return_vecs
        self.bert_model = AutoModel.from_pretrained(cfg.bert_model)
        for p in self.bert_model.parameters():
            p.requires_grad = cfg.trainable
        self._dropout = torch.nn.Dropout(p=cfg.dropout)
        self.compressor = torch.nn.Linear(self.bert_model.config.hidden_size, cfg.compression_dim)

    def forward(self,
                query: Dict[str, torch.LongTensor],
                document: Dict[str, torch.LongTensor],
                use_fp16:bool = True, 
                output_secondary_output: bool = False) -> torch.Tensor:
                
        with torch.cuda.amp.autocast(enabled=use_fp16):

            query_vecs = self.forward_representation(query)
            document_vecs = self.forward_representation(document)

            score_per_term = torch.bmm(query_vecs, document_vecs.transpose(2, 1))
            score_per_term[~(document["attention_mask"].bool()).unsqueeze(1).expand(-1,score_per_term.shape[1],-1)] = - 1000
            score = score_per_term.max(-1).values
            score[~(query["attention_mask"].bool())] = 0
            score = score.sum(-1)

            if self.is_teacher_model:
                return (score, query_vecs, document_vecs)

            if self.return_vecs:
                score = (score, query_vecs, document_vecs)

            if output_secondary_output:
                return score, {}
            return score

    def forward_representation(self,  # type: ignore
                               tokens: Dict[str, torch.LongTensor],
                               sequence_type=None) -> Dict[str, torch.Tensor]:
        
        vecs = self.bert_model(**tokens)[0]
        vecs = self.compressor(vecs)

        if sequence_type == "doc_encode" or sequence_type == "query_encode":
            vecs = vecs * tokens["attention_mask"].unsqueeze(-1)

        return vecs

    def forward_aggregation(self, query_vecs, document_vecs):
        score = torch.bmm(query_vecs, document_vecs.transpose(2,1))
        score = score.max(-1).values
        score = score.sum(-1)
        return score

    def forward_inbatch_aggregation(self, query_vecs, query_mask, document_vecs, document_mask):
        score = torch.mm(query_vecs.view(-1,query_vecs.shape[-1]), document_vecs.view(-1, document_vecs.shape[-1]).transpose(-2,-1))\
                     .view(query_vecs.shape[0],query_vecs.shape[1],document_vecs.shape[0],document_vecs.shape[1])
        score=score.transpose(1, 2)

        score[~(document_mask.bool()).unsqueeze(1).unsqueeze(1).expand(-1,score.shape[1],score.shape[2],-1)] = - 1000
        score = score.max(-1).values
        score[~(query_mask.bool()).unsqueeze(1).expand(-1,score.shape[1],-1)] = 0
        score = score.sum(-1)
        return score

    def get_param_stats(self):
        return "ColBERT: / "


    def get_param_secondary(self):
        return {}