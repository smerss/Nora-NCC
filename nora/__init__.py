"""Nora: A living, continually learning, biologically inspired cognitive architecture.

Modular Architecture Subpackages:
    nora.substrate: Physical simulation, connectome, LIF dynamics, neurochemistry
    nora.cognition: Unified mind, global workspace, cognitive bus, AGI architecture
    nora.self_model: Homeostasis, self-model, inner monologue, drives, metacognition
    nora.reasoning: Deliberation, causal inference, planning, logic
    nora.memory: Epistemic store, episodic memory, schemas, priors
    nora.language: Turkish linguistic stack, Broca articulation, discourse
    nora.synthesis: Code synthesis, skill forge, curriculum, evolutionary learning
    nora.models: Neural networks, bridge thought columns, growing transformers
"""

__version__ = "0.1.0"

# Install legacy import compatibility hook
from . import _compat
_compat.install()


def __getattr__(name):
    # Lazy attribute loading for primary convenient constructs
    if name == "Mind":
        from .cognition.mind import Mind
        return Mind
    if name == "Universe":
        from .substrate.world import Universe
        return Universe
    if name == "live":
        from .self_model.life import live
        return live
    if name == "NoraAGI":
        from .cognition.agi import NoraAGI
        return NoraAGI
    raise AttributeError(f"module 'nora' has no attribute '{name}'")
