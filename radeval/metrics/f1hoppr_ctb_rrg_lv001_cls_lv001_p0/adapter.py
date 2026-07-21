from ..f1hoppr_shared.adapter import F1HopprMetricAdapter
from .metric import HopprF1CtbRrgLv001ClsLv001P0


class F1HopprCtbRrgLv001ClsLv001P0Metric(F1HopprMetricAdapter):
    name = "f1hoppr_ctb_rrg_lv001_cls_lv001_p0"
    display_name = "F1HopprCtbRrgLv001ClsLv001P0"
    scorer_class = HopprF1CtbRrgLv001ClsLv001P0
