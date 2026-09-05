# init, AriaQuanta.qml
#
# Layout, and where things go as QML support keeps growing (see the project progress
# report for the underlying math):
#
#   qml/
#     _shared.py        <- Items shared between other files and folders, such as validations and 
#                          Pauli matrices and the resolve_qubit_subsets function in Connectivity
#     encoding/         <- plain classical->quantum data maps, NO trainable parameters
#                          (basis, amplitude, angle, hamiltonian)
#     feature_map/      <- Havlicek-style structured encodings (Pauli/Z/ZZ/IQP), still data-only/no trainable 
#                          parameters, but distinguished by their specific "repeated H + diagonal Pauli evolution" 
#                          structure and their usual purpose (quantum kernels)
#     ansatz/           <- trainable/variational circuits (hardware-efficient, data
#                          re-uploading -- the latter is a hybrid: data AND weights)
#     circuits/         <- general PQC infrastructure                           (not yet implemented)
#     gradients/        <- parameter-shift rule and other gradient estimators   (not yet implemented)
#     training/         <- hybrid quantum-classical optimization loops          (not yet implemented)
#     models/           <- ready-made models such as VQC                        (not yet implemented)
#
# Each future sub-package gets its own __init__.py exporting its public names, mirroring how
# AriaQuanta.aqc / AriaQuanta.algorithms / AriaQuanta.backend are already organized.

from .encoding import (
    basis_encoding,
    amplitude_encoding,
    angle_encoding,
    hamiltonian_encoding,
    hamiltonian_matrix,
)

from .feature_map import (
    pauli_feature_map,
    z_feature_map,
    zz_feature_map,
    iqp_feature_map,
)

from .ansatz import (
    HardwareEfficientAnsatz,
    DataReUploadingAnsatz,
)
