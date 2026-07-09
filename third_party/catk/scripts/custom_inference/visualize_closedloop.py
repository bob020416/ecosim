"""CATK closed-loop ECoSim inference entry point."""

import utils_closed_loop as utils


def main() -> None:
    args = utils.parse_args()
    utils.run_visualization(args)


if __name__ == "__main__":
    main()
