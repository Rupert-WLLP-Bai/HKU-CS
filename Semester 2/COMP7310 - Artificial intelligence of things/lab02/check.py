# --------------------------------------------------------------------------------
#
# WARNING: This file checks your solution and zips the files for submission.
# Please DO NOT CHANGE ANY PART of this file unless you are absolutely sure of
# the consequences and have consulted with the TA.
#
# --------------------------------------------------------------------------------


from rich import print as rprint
from rich.console import Console
import builtins
builtins.print = rprint
console = Console()

import argparse
import os
from datetime import datetime
import zipfile
import subprocess
import sys

LAB_ID = 2
TASK_ID = [1, 2, 3, 4]

SOL = False # DO NOT CHANGE THIS
suffix = "_sol" if SOL else ""

if SOL:
    # Import task modules
    from task_2_1_sol import task_2_1
    from task_2_2_sol import task_2_2
    from task_2_3_sol import task_2_3
    from task_2_4_sol import task_2_4
else:
    # Import task modules
    from task_2_1 import task_2_1
    from task_2_2 import task_2_2
    from task_2_3 import task_2_3
    from task_2_4 import task_2_4

def run_pytest_tests(task):
    """
    Run pytest for the specified task.

    Parameters
    ----------
    task : int
        The task number (1, 2, 3 or 4).
    """
    console.print(f"\n[bold cyan]********* Running Tests for Task {LAB_ID}-{task} *********[/bold cyan]")
    
    # Define the test file based on the task
    test_file = f"task_{LAB_ID}_{task}{suffix}.py"
    
    # Run pytest with doctest support and verbose output
    result = subprocess.run(
        ["pytest", "--doctest-modules", "-v", test_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Print colorful output
    console.print(result.stdout)
    if result.returncode != 0:
        console.print(f"\n[bold red]❌ Some tests failed in Task {LAB_ID}-{task}![/bold red]")
        return False
    else:
        console.print(f"\n[bold green]✅ All tests passed in Task {LAB_ID}-{task}![/bold green]")
        return True

def run_pytest_tests_all():
    """
    Run pytest for all tasks.
    """
    console.print(f"\n[bold cyan]********* Running Tests for Task {LAB_ID}*********[/bold cyan]")
    test_files = []
    for task in TASK_ID:
        test_files.append(f"task_{LAB_ID}_{task}{suffix}.py")
    flag = True
    for test_file in test_files:
        result = subprocess.run(
            ["pytest", "--doctest-modules", "-v", test_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        console.print(result.stdout)
        if result.returncode != 0:
            console.print(f"\n[bold red]❌ Some tests failed in {test_file}![/bold red]")
            flag = False
    return flag

def zip_files(uid):
    """
    Zip the submission files, including the `fig` folder.

    Parameters
    ----------
    uid : str
        The user's university ID.
    """
    console.print(f"\n[bold cyan]********* Zipping Files *********[/bold cyan]")
    
    # Files and folders to include in the zip
    submit_files = [f"task_2_1{suffix}.py", 
                    f"task_2_2{suffix}.py", 
                    f"task_2_3{suffix}.py",
                    f"task_2_4{suffix}.py",
                    f"answer_sheet.pickle"]
    submit_folders = []  # Folder containing figures
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    zip_file_name = f"{uid}_lab_{LAB_ID}_{timestamp}.zip"

    with zipfile.ZipFile(zip_file_name, "w") as zip_file:
        # Add Python files
        for file in submit_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"File {file} not found in {os.getcwd()}")
            console.print(f"Zipping {file}...", end=" ")
            zip_file.write(file)
            console.print("[green]Done.[/green]")
        
        # Add the `fig` folder and its contents
        for folder in submit_folders:
            if not os.path.exists(folder):
                raise FileNotFoundError(f"Folder {folder} not found in {os.getcwd()}")
            for root, dirs, files in os.walk(folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    console.print(f"Zipping {file_path}...", end=" ")
                    zip_file.write(file_path, os.path.relpath(file_path, start="."))
                    console.print("[green]Done.[/green]")
    
    return zip_file_name
    

def collect_solution_1(data_dict):
    print(f"Collecting the solution for task 2_1 ...", end="\t")
    test_1 = task_2_1()
    try:
        freq1, mag1 = test_1.apply_fft_pt()
    except:
        raise Exception("Please check your solution for task_2_1.")
    
    data_dict["task_2_1"] = {
        "freq": freq1[-1],
        "mag": mag1[-1]
    }
    print(f"[green]Done.[/green]")
    return data_dict

def collect_solution_2(data_dict):
    print(f"Collecting the solution for task 2_2 ...", end="\t")
    test_2 = task_2_2()
    try:
        freq1 = test_2.get_freq_spt()
        bw2 = test_2.get_bw_chirp()
        hr3 = test_2.get_heart_rate()
    except:
        raise Exception("Please check your solution for task_2_2.")
    
    data_dict["task_2_2"] = {
        "2_2_1": {
            "freq": freq1
        },
        "2_2_2": {
            "bw": bw2
        },
        "2_2_3": {
            "hr": hr3
        },
    }
    print(f"[green]Done.[/green]")
    return data_dict

def collect_solution_3(data_dict):
    print(f"Collecting the solution for task 2_3 ...", end="\t")
    test_3 = task_2_3()
    try:
        fs_1, N_1, f_1 = test_3.get_freq_1()
        fs_2, N_2, f_2 = test_3.get_freq_2()
    except:
        raise Exception("Please check your solution for task_2_3.")
    data_dict["task_2_3"] = {
        "2_3_1": {
            "fs": fs_1,
            "N": N_1,
            "f": f_1
        },
        "2_3_2": {
            "fs": fs_2,
            "N": N_2,
            "f": f_2
        }
    }
    print(f"[green]Done.[/green]")
    return data_dict

def collect_solution_4(data_dict):
    print(f"Collecting the solution for task 2_4 ...", end="\t")
    test_4 = task_2_4()
    try:
        tx_1 = test_4.generate_transmitted_signal()
        if_2 = test_4.compute_if_signal()
        d_3, rfft_3, rbins_3 = test_4.estimate_distance()
        aoas = test_4.estimate_AoA()
    except:
        raise Exception("Please check your solution for task_2_4.")
    data_dict["task_2_4"] = {
        "2_4_1": {
            "tx": tx_1[-1]
        },
        "2_4_2": {
            "if": if_2[-1]
        },
        "2_4_3": {
            "d": d_3,
            "rfft": rfft_3,
            "rbins": rbins_3
        },
        "2_4_4": {
            "aoas": aoas
        }
    }
    print(f"[green]Done.[/green]")
    return data_dict

def save_to_pickle(data_dict):
    import pickle
    with open("answer_sheet.pickle", "wb") as f:
        pickle.dump(data_dict, f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check the solution and zip the files.')
    parser.add_argument("--uid", type=str, help="Your University ID. e.g. 1234567")
    parser.add_argument("--task", type=int, choices=[1, 2, 3, 4], help="The task number to check (1, 2, or 3).")
    args = parser.parse_args()

    # Run tests and visualize for the specified task
    if args.task:
        success = run_pytest_tests(args.task)
        
        if not success:
            console.print("\n[bold yellow]⚠️  Fix the test failures.[/bold yellow]")
            sys.exit(1)

    # Zip files if UID is provided
    if args.uid:
        # run all tests
        data_dict = {}
        res = run_pytest_tests_all()
        
        if res:
            console.print("\n[bold green]✅ All tests passed![/bold green]")
            data_dict = collect_solution_1(data_dict)
            data_dict = collect_solution_2(data_dict)
            data_dict = collect_solution_3(data_dict)
            data_dict = collect_solution_4(data_dict)
            save_to_pickle(data_dict)
    
        else:
            console.print("\n[bold yellow]⚠️  Fix the test failures before zipping.[/bold yellow]")
            sys.exit(1)
        
        console.print(f"\nYour UID is [bold]{args.uid}[/bold]. Is this correct? (y/n): ", end="")
        if input().lower() != "y":
            console.print("[yellow]Zip operation canceled.[/yellow]")
            sys.exit(0)
        
        try:
            zip_name = zip_files(args.uid)
            console.print(f"\n[bold green]📦 Successfully created submission package: {zip_name}[/bold green]")
            console.print("[bold green]Please submit this file to Moodle.[/bold green]")
        except Exception as e:
            console.print(f"\n[bold red]❌ Error during zipping: {str(e)}[/bold red]")
            sys.exit(1)

