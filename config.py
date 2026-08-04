from typing import Any, Optional

import numpy as npnumpy
try:
    import cupy as npcupy
    cupy_exist = True
except ImportError:
    npcupy = None
    cupy_exist = False


VALID_HARDWARE = ('Local', 'HPC', 'Cloud', 'QPU')


# -------------------------------------------------------------------------------------------
class Config:
    hardware: str = 'Local'
    use_gpu: bool = False

    @staticmethod
    def set_use_gpu(this_use_gpu: bool) -> None:
        if this_use_gpu and cupy_exist:
            print('use_gpu is set to True. Running on GPU.')
        elif this_use_gpu and not cupy_exist:
            this_use_gpu = False
            print('use_gpu is set to True, but cupy is not installed. Falling back to CPU (use_gpu stays False).')
        else:
            print('use_gpu is set to False. Running on CPU.')

        Config.use_gpu = this_use_gpu

    @staticmethod
    def set_hardware(this_hardware: str) -> None:
        if this_hardware not in VALID_HARDWARE:
            raise ValueError("'hardware' must be one of {}, got {!r}.".format(VALID_HARDWARE, this_hardware))
        if this_hardware != 'Local':
            print("Note: hardware={!r} is declared but not implemented yet -- Simulator currently only runs 'Local'.".format(this_hardware))
        Config.hardware = this_hardware

    @staticmethod
    def gpu_info() -> str:
        # a short, human-readable summary of GPU availability/status -- print(Config.gpu_info())
        if not cupy_exist:
            return "cupy is not installed -- GPU acceleration is unavailable (running on CPU)."

        try:
            device_id = npcupy.cuda.Device().id
            properties = npcupy.cuda.runtime.getDeviceProperties(device_id)
            name = properties['name']
            name = name.decode() if isinstance(name, bytes) else name
            free_bytes, total_bytes = npcupy.cuda.runtime.memGetInfo()
            free_gb, total_gb = free_bytes / (1024 ** 3), total_bytes / (1024 ** 3)
            status = 'active' if Config.use_gpu else 'available but not active (Config.use_gpu is False)'
            return "cupy is installed; GPU {} ({}): {:.2f} / {:.2f} GB free -- {}.".format(device_id, name, free_gb, total_gb, status)
        except Exception as error:
            return "cupy is installed, but querying the GPU failed: {}.".format(error)

# -------------------------------------------------------------------------------------------
class PrintOptions:
    precision: int = 4
    suppress_small: bool = True
    linewidth: int = 120

    @staticmethod
    def apply() -> None:
        npnumpy.set_printoptions   (precision=PrintOptions.precision, suppress=PrintOptions.suppress_small, linewidth=PrintOptions.linewidth)
        if cupy_exist:
            npcupy.set_printoptions(precision=PrintOptions.precision, suppress=PrintOptions.suppress_small, linewidth=PrintOptions.linewidth)

    @staticmethod
    def set(precision: Optional[int] = None, suppress_small: Optional[bool] = None, linewidth: Optional[int] = None) -> None:
        if precision is not None:
            if precision < 0:
                raise ValueError("'precision' must be non-negative, got {}.".format(precision))
            PrintOptions.precision = precision

        if suppress_small is not None:
            PrintOptions.suppress_small = bool(suppress_small)

        if linewidth is not None:
            if linewidth < 1:
                raise ValueError("'linewidth' must be positive, got {}.".format(linewidth))
            PrintOptions.linewidth = linewidth

        PrintOptions.apply()

PrintOptions.apply()   # so the defaults above take effect as soon as AriaQuanta is imported
    
# -------------------------------------------------------------------------------------------
def get_array_module(this_use_gpu: bool) -> Any:
    if this_use_gpu and (npcupy is not None): return npcupy
    else: return npnumpy


