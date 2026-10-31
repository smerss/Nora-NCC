"""CLI runner for Nora tools: python -m tools [tool_name] [args...]"""
import importlib
import runpy
import sys

TOOL_MAP = {
    "make_corpus": "tools.make_corpus",
    "make_corpus_dump": "tools.make_corpus_dump",
    "make_corpus_tr": "tools.make_corpus_tr",
    "make_fly_reservoir": "tools.make_fly_reservoir",
    "make_think": "tools.make_think",
    "make_think1": "tools.make_think1",
    "make_tr": "tools.make_tr",
    "probe_digit": "tools.probe_digit",
    "probe_feat": "tools.probe_feat",
    "tool_calc": "tools.tool_calc",
    "calc": "tools.tool_calc",
    "memcheck": "tools.memcheck",
    "merdiven": "tools.merdiven",
    "gen_seg": "tools.gen_seg",
    "benchmark": "tools.benchmark_profile",
    "benchmark_profile": "tools.benchmark_profile",
    "compare_broca": "tools.compare_broca_decoders",
    "compare_broca_decoders": "tools.compare_broca_decoders",
}


def print_help():
    print("Nora Tool Runner")
    print("Usage: python -m tools <tool_name> [args...]")
    print("\nAvailable tools:")
    for k in sorted(TOOL_MAP.keys()):
        print(f"  - {k:22} ({TOOL_MAP[k]})")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help", "list"):
        print_help()
        sys.exit(0)
    name = sys.argv[1]
    mod_name = TOOL_MAP.get(name)
    if not mod_name:
        print(f"Unknown tool: {name}")
        print_help()
        sys.exit(1)

    sys.argv = [mod_name] + sys.argv[2:]
    runpy.run_module(mod_name, run_name="__main__")


if __name__ == "__main__":
    main()
