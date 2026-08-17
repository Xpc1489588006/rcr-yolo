"""Register RCR-YOLO custom modules into Ultralytics so yaml configs work.

Ultralytics' ``parse_model`` (ultralytics/nn/tasks.py) resolves unknown module
class names through the module globals of ``ultralytics.nn.tasks``; injecting
the classes there is the community-standard, non-invasive extension point
(no edits to the installed package).

Usage:
    from rcr.ultralytics_patch import register_rcr_modules
    register_rcr_modules()
    model = YOLO("cfg/yolo11n-rcr.yaml")
"""

from . import GSConv, LCR, LCRBase, MRFE, ORBIn

_RCRC_CLASSES = (GSConv, LCR, LCRBase, MRFE, ORBIn)


def register_rcr_modules():
    """Inject RCR module classes into ultralytics.nn.tasks namespace."""
    import ultralytics.nn.tasks as tasks

    for cls in _RCRC_CLASSES:
        setattr(tasks, cls.__name__, cls)
    return tasks
