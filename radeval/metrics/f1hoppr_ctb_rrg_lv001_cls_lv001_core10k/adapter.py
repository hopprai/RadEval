from ..f1hoppr_shared.adapter import F1HopprMetricAdapter
from .metric import HopprF1CtbRrgLv001ClsLv001Core10k


class F1HopprCtbRrgLv001ClsLv001Core10kMetric(F1HopprMetricAdapter):
    name = "f1hoppr_ctb_rrg_lv001_cls_lv001_core10k"
    display_name = "F1HopprCtbRrgLv001ClsLv001Core10k"
    scorer_class = HopprF1CtbRrgLv001ClsLv001Core10k
