"""47-head core10k CT-body metric for ctb_rrg_lv001 + ctb_cls_lv001."""
from ..f1hoppr_shared.ctb_scopes import CORE10K
from ..f1hoppr_shared.scorer import HopprMultiOutputScorer

_CKPT = (
    "/nfs/cluster/hoppr_vlm_ressources/radeval_checkpoints/"
    "f1hoppr_ctb_rrg_lv001_cls_lv001_core10k"
)


class HopprF1CtbRrgLv001ClsLv001Core10k(HopprMultiOutputScorer):
    def __init__(self, checkpoint_dir=_CKPT, **kwargs):
        kwargs.setdefault("max_length", 512)
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            expected_conditions=CORE10K,
            **kwargs,
        )
