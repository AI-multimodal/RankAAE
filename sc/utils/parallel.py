import os
import argparse


def ipcluster(action, n=8, ipypar_path='.ipypar'):
    """
    Start and stop ipcluster.
    """
    assert action in ['start', 'stop']
    
    if action == 'start':
        command = ' '.join(
            [
                "ipcluster", "start", f"-n={n}", f"--profile-dir={ipypar_path}"
            ]
        )
    else:
        command = ' '.join(
        [
            "ipcluster", "stop", f"--profile-dir={ipypar_path}"
        ]
    )

    os.system(command)



def kill_ipypar_pcocesses(user='zliang'):
    """
    If stop cluster failed, run this command.
    """
    command = ' | '.join(
        [
            "ps -ef",
            f"grep {user}",
            "grep ipyp",
            "awk '{print $2}'",
            "~/parallel/bin/parallel -j 1 'kill -9 {}'"
        ]
    )
        
    os.system(command)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', type=str, default='stop',
                        help="Action should be one of these: start, stop, kill.")
    parser.add_argument('-n', '--num_engine', type=int, default=8)
    parser.add_argument('-p', '--profile-dir', type=str, default='./ipypar')
    args = parser.parse_args()
    
    if args.action != "kill":
        ipcluster(args.action, n=args.num_engine, ipypar_path=args.profile_dir)
    else:
        kill_ipypar_pcocesses(user='zliang')