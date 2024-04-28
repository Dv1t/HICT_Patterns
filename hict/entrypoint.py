import argparse
import sys
from typing import Callable, Dict
from hict.patterns import hict_patterns
from hict.patterns import visualize_sv


commands_entrypoints: Dict[str, Callable] = {
    'pattern_search': hict_patterns.main,
    'visualize_sv': visualize_sv.main
}


def main():
    parser = argparse.ArgumentParser(
        description="HiCT utilities package", prefix_chars="-+",
        epilog="Visit https://github.com/Dv1t/HICT_Patterns for more info."
    )
    parser.add_argument(
        "tool",
        help="Tool to run",
        choices=commands_entrypoints.keys(),
    )
    args = parser.parse_args(sys.argv[1:2])
    subroutine: Callable
    try:
        subroutine: Callable = commands_entrypoints[args.tool]
    except KeyError:
        print(f"Unrecognized tool: {args.tool}")
        parser.print_help()
        exit(1)
    subroutine(sys.argv[2:])


if __name__ == '__main__':
    main()