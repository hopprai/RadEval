"""87-head full CT-body metric for ctb_rrg_lv001 + ctb_cls_lv001."""
from ..f1hoppr_shared.ctb_scopes import ALL
from ..f1hoppr_shared.scorer import HopprMultiOutputScorer

_CKPT = (
    "/nfs/cluster/hoppr_vlm_ressources/radeval_checkpoints/"
    "f1hoppr_ctb_rrg_lv001_cls_lv001_all"
)


class HopprF1CtbRrgLv001ClsLv001All(HopprMultiOutputScorer):
    def __init__(self, checkpoint_dir=_CKPT, **kwargs):
        kwargs.setdefault("max_length", 512)
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            expected_conditions=ALL,
            **kwargs,
        )
