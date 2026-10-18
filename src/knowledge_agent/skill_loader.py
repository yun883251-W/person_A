"""兼容导入入口；实现位于 knowledge_agent.skills.loader。"""
from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("knowledge_agent.skills.loader")
