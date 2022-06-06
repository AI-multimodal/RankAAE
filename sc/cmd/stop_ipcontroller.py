import argparse, os
import ipyparallel as ipp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', "--work_dir", type=str, default='.',
                        help="Working directory to write the output files")
    args = parser.parse_args()
    work_dir = os.path.abspath(os.path.expanduser(args.work_dir))

    c = ipp.Client(connection_info=f"{work_dir}/ipypar/security/ipcontroller-client.json")
    c.shutdown(hub=True)
    
if __name__ == '__main__':
    main()
