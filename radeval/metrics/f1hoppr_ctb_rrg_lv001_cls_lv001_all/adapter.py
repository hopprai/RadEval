from ..f1hoppr_shared.adapter import F1HopprMetricAdapter
from .metric import HopprF1CtbRrgLv001ClsLv001All


class F1HopprCtbRrgLv001ClsLv001AllMetric(F1HopprMetricAdapter):
    name = "f1hoppr_ctb_rrg_lv001_cls_lv001_all"
    display_name = "F1HopprCtbRrgLv001ClsLv001All"
    scorer_class = HopprF1CtbRrgLv001ClsLv001All
