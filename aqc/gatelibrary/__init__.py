
# init, AriaQuanta.aqc.gatelibrary

#------------------------------------------------------------------------------------
from .gatesingle import *
"""
    GateSingleQubit,
    I,
    GlobalPhase,
    X,
    Y,
    Z,
    S,
    Xsqrt,
    H,
    P,
    T,
    RX, # supports params
    RY, # supports params
    RZ, # supports params
    Rot,
"""

#------------------------------------------------------------------------------------
from .gatedouble import *
"""
    GateDoubleQubit,
    SWAP,
    ISWAP,
    SWAPsqrt,
    SWAPalpha,
    RXX,  # supports params
    RYY,  # supports params
    RZZ,
    RXY,
    Barenco,
    Berkeley,
    Canonical,
    Givens,
    Magic,
"""

#------------------------------------------------------------------------------------
from .arxived_gatetriple import *
"""
    GateTripleQubit,
    CCXold,
    CSWAPold,
"""

#------------------------------------------------------------------------------------
from .gatecustom import *
"""
    GateCustom,
    Custom,
"""

#------------------------------------------------------------------------------------
from .gatecontrol import *
"""
    GateControl,
    CX,
    CZ,
    CP,   # supports params
    CS,
    CSX, 
    CRX,  # supports params  
    CCX, 
    CSWAP,
    CU,
"""

#------------------------------------------------------------------------------------
from .gatecontroln import *
"""
    CNZ
"""