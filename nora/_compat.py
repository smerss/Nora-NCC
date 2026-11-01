"""Backward-compatibility module mapping for Nora.

Ensures that legacy imports like `import nora_agi` or `from nora_world import Universe`
automatically resolve to their modular package counterparts without requiring flat root files.
"""
import importlib
import importlib.abc
import importlib.util
import sys

MODULE_MAP = {
    # Substrate
    "nora_world": "nora.substrate.world",
    "nora_neuron": "nora.substrate.neuron",
    "nora_chem": "nora.substrate.chem",
    "nora_nervous": "nora.substrate.nervous",
    "nora_entropy": "nora.substrate.entropy",
    "nora_state": "nora.substrate.state",
    "nora_crossmodal": "nora.substrate.crossmodal",
    # Cognition
    "nora_mind": "nora.cognition.mind",
    "nora_agi": "nora.cognition.agi",
    "nora_bus": "nora.cognition.bus",
    "nora_gw": "nora.cognition.gw",
    "nora_architecture": "nora.cognition.architecture",
    "nora_workspace": "nora.cognition.workspace",
    # Self-Model
    "nora_self": "nora.self_model.self",
    "nora_life": "nora.self_model.life",
    "nora_life_experiment": "nora.self_model.life_experiment",
    "nora_drive": "nora.self_model.drive",
    "nora_initiative": "nora.self_model.initiative",
    "nora_monologue": "nora.self_model.monologue",
    "nora_metacog": "nora.self_model.metacog",
    "nora_meta": "nora.self_model.meta",
    "nora_selfexam": "nora.self_model.selfexam",
    "nora_mirror": "nora.self_model.mirror",
    # Reasoning
    "nora_deliberation": "nora.reasoning.deliberation",
    "nora_planning": "nora.reasoning.planning",
    "nora_causal": "nora.reasoning.causal",
    "nora_prediction": "nora.reasoning.prediction",
    "nora_speculation": "nora.reasoning.speculation",
    "nora_infer": "nora.reasoning.infer",
    "nora_autonomous_logic": "nora.reasoning.logic",
    "nora_laws": "nora.reasoning.laws",
    "nora_sets": "nora.reasoning.sets",
    # Memory
    "nora_episteme": "nora.memory.episteme",
    "nora_episodic": "nora.memory.episodic",
    "nora_store": "nora.memory.store",
    "nora_priors": "nora.memory.priors",
    "nora_provenance": "nora.memory.provenance",
    "nora_knowledge": "nora.memory.knowledge",
    "nora_objects": "nora.memory.objects",
    "nora_frames": "nora.memory.frames",
    "nora_schema": "nora.memory.schema",
    "nora_reify": "nora.memory.reify",
    "nora_abstract": "nora.memory.abstract",
    # Language
    "nora_lang": "nora.language.lang",
    "nora_turkce": "nora.language.turkce",
    "nora_broca": "nora.language.broca",
    "nora_broca_net": "nora.language.broca_net",
    "nora_discourse": "nora.language.discourse",
    "nora_voice": "nora.language.voice",
    "nora_ear": "nora.language.ear",
    "nora_anlatim": "nora.language.anlatim",
    "nora_style": "nora.language.style",
    "nora_translation": "nora.language.translation",
    "nora_register": "nora.language.register",
    # Synthesis
    "nora_synth": "nora.synthesis.synth",
    "nora_forge": "nora.synthesis.forge",
    "nora_apply": "nora.synthesis.apply",
    "nora_code": "nora.synthesis.code",
    "nora_semantic_code": "nora.synthesis.semantic_code",
    "nora_evolution": "nora.synthesis.evolution",
    "nora_stem": "nora.synthesis.stem",
    "nora_studio": "nora.synthesis.studio",
    "nora_okul": "nora.synthesis.okul",
    "nora_besle": "nora.synthesis.besle",
    "nora_egit": "nora.synthesis.egit",
    "nora_audit": "nora.synthesis.audit",
    "nora_sayfa": "nora.synthesis.sayfa",
    # Models
    "nora_bridge": "nora.models.bridge",
    "nora_think": "nora.models.think",
    "micro_gpt": "nora.models.micro_gpt",
    "grow_gpt": "nora.models.grow_gpt",
    "growing_net": "nora.models.growing_net",
    # Batteries
    "agi_battery": "batteries.agi_battery",
    "architecture_battery": "batteries.architecture_battery",
    "battery": "batteries.battery",
    "bridge_battery": "batteries.bridge_battery",
    "causal_battery": "batteries.causal_battery",
    "consciousness_battery": "batteries.consciousness_battery",
    "crossmodal_battery": "batteries.crossmodal_battery",
    "evolution_battery": "batteries.evolution_battery",
    "external_battery": "batteries.external_battery",
    "heldout_battery": "batteries.heldout_battery",
    "heldout_negatif": "batteries.heldout_negatif",
    "heldout_sealed": "batteries.heldout_sealed",
    "heldout_sealed2": "batteries.heldout_sealed2",
    "heldout_sealed3": "batteries.heldout_sealed3",
    "integrate_battery": "batteries.integrate_battery",
    "law_battery": "batteries.law_battery",
    "metacog_battery": "batteries.metacog_battery",
    "object_battery": "batteries.object_battery",
    "persistence_battery": "batteries.persistence_battery",
    "planning_battery": "batteries.planning_battery",
    "practical_language_battery": "batteries.practical_language_battery",
    "survival_battery": "batteries.survival_battery",
    "translation_battery": "batteries.translation_battery",
    "workspace_battery": "batteries.workspace_battery",
    "selftest": "batteries.selftest",
    "solo_gorev": "batteries.solo_gorev",
    "sozce_siniflandirma_degerlendirme": "batteries.sozce_siniflandirma_degerlendirme",
    # Tools
    "tool_calc": "tools.tool_calc",
    "make_corpus": "tools.make_corpus",
    "make_corpus_dump": "tools.make_corpus_dump",
    "make_corpus_tr": "tools.make_corpus_tr",
    "make_fly_reservoir": "tools.make_fly_reservoir",
    "make_think": "tools.make_think",
    "make_think1": "tools.make_think1",
    "make_tr": "tools.make_tr",
    "probe_digit": "tools.probe_digit",
    "probe_feat": "tools.probe_feat",
    "memcheck": "tools.memcheck",
    "merdiven": "tools.merdiven",
    "gen_seg": "tools.gen_seg",
    "benchmark_profile": "tools.benchmark_profile",
}


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target_name):
        self.target_name = target_name

    def create_module(self, spec):
        mod = importlib.import_module(self.target_name)
        sys.modules[spec.name] = mod
        return mod

    def exec_module(self, module):
        pass


class NoraCompatFinder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self._busy = set()

    def find_spec(self, fullname, path, target=None):
        if fullname in MODULE_MAP and fullname not in self._busy:
            self._busy.add(fullname)
            try:
                target_name = MODULE_MAP[fullname]
                return importlib.util.spec_from_loader(fullname, _AliasLoader(target_name))
            finally:
                self._busy.discard(fullname)
        return None


_finder_instance = NoraCompatFinder()


def install():
    if not any(isinstance(f, NoraCompatFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _finder_instance)
