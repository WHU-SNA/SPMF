from spmf import SPMFModel
from utils import parameter_parser


def main():
    args = parameter_parser()
    SPMF = SPMFModel(args)
    SPMF.calculate()


if __name__ == "__main__":
    main()