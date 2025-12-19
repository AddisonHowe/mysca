import sys

def main():
    from mysca.main import parse_args, main
    args = parse_args(sys.argv[1:])
    main(args)

def run_sca():
    from mysca.run_sca import parse_args, main
    args = parse_args(sys.argv[1:])
    main(args)

if __name__ == "__main__":
    main()
