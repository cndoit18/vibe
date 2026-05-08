import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

# langgraph 1.1.x 在 import 时触发 Reviver() 无参构造的 pending deprecation warning
# 这是上游 bug，尚未修复，在此全局过滤
warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
