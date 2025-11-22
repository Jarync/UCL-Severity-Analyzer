# torchlm_patch.py
import scipy.integrate as _si
if not hasattr(_si, "simps") and hasattr(_si, "simpson"):
    _si.simps = _si.simpson
