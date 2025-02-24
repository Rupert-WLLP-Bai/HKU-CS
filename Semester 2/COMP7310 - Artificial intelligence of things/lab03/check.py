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
import pickle

LAB_ID = 3
TASK_ID = [1, 2, 3]

SOL = False  # DO NOT CHANGE THIS
suffix = "_sol" if SOL else ""

if SOL:
    # Import task modules
    from task_3_1_sol import task_3_1
    from task_3_2_sol import task_3_2
    from task_3_3_sol import task_3_3
else:
    # Import task modules
    from task_3_1 import task_3_1
    from task_3_2 import task_3_2
    from task_3_3 import task_3_3

def run_pytest_tests(task):
    """
    Run pytest for the specified task.

    Parameters
    ----------
    task : int
        The task number (1, 2, or 3).
    """
    console.print(f"\n[bold cyan]********* Running Tests for Task {LAB_ID}-{task} *********[/bold cyan]")
    
    test_file = f"task_{LAB_ID}_{task}{suffix}.py"
    
    result = subprocess.run(
        ["pytest", "--doctest-modules", "-v", test_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
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
    console.print(f"\n[bold cyan]********* Running Tests for Task {LAB_ID} *********[/bold cyan]")
    test_files = [f"task_{LAB_ID}_{task}{suffix}.py" for task in TASK_ID]
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
    Zip the submission files.

    Parameters
    ----------
    uid : str
        The user's university ID.
    """
    console.print(f"\n[bold cyan]********* Zipping Files *********[/bold cyan]")
    
    submit_files = [
        f"task_3_1{suffix}.py",
        f"task_3_2{suffix}.py",
        f"task_3_3{suffix}.py",
        "answer_sheet.pickle"
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    zip_file_name = f"{uid}_lab_{LAB_ID}_{timestamp}.zip"

    with zipfile.ZipFile(zip_file_name, "w") as zip_file:
        for file in submit_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"File {file} not found in {os.getcwd()}")
            console.print(f"Zipping {file}...", end=" ")
            zip_file.write(file)
            console.print("[green]Done.[/green]")
    
    return zip_file_name

def collect_solution_1(data_dict):
    """
    Collect solutions for Task 3-1.
    """
    print(f"Collecting the solution for task 3_1 ...", end="\t")
    test_1 = task_3_1()
    try:
        acf1 = test_1.apply_acf_pt()
        acf2 = test_1.apply_acf_pulse()
    except:
        raise Exception("Please check your solution for task_3_1.")
    
    data_dict["task_3_1"] = {
        "3_1_1": {"acf": acf1},
        "3_1_2": {"acf": acf2}
    }
    print(f"[green]Done.[/green]")
    return data_dict

def collect_solution_2(data_dict):
    """
    Collect solutions for Task 3-2.
    """
    print(f"Collecting the solution for task 3_2 ...", end="\t")
    test_2 = task_3_2()
    try:
        br1 = test_2.get_br_1()
        br2 = test_2.get_br_2()
        hr1 = test_2.get_hr_1()
        hr2 = test_2.get_hr_2()
    except:
        raise Exception("Please check your solution for task_3_2.")
    
    data_dict["task_3_2"] = {
        "3_2_1": {"br": br1},
        "3_2_2": {"br": br2},
        "3_2_3": {"hr": hr1},
        "3_2_4": {"hr": hr2}
    }
    print(f"[green]Done.[/green]")
    return data_dict

def collect_solution_3(data_dict):
    """
    Collect solutions for Task 3-3.
    """
    print(f"Collecting the solution for task 3_3 ...", end="\t")
    test_3 = task_3_3()
    try:
        x1 = [1, 2, 3]
        x2 = [2, 4, 6]
        x3 = [3, 2, 1]
        pcc1, pcc2 = test_3.get_pcc(x1, x2), test_3.get_pcc(x1, x3)
        pcc_m, motion_ok = test_3.check_motion_sensors(test_3.m1, test_3.m2)
        pcc_t, temp_ok = test_3.check_temperature_sensors(test_3.t1, test_3.t2)
        delay, alarm_ok = test_3.sync_event_signals(test_3.s1, test_3.s2)
        music_idx = test_3.detect_music_patterns(test_3.s, test_3.p)
    except:
        raise Exception("Please check your solution for task_3_3.")
    
    data_dict["task_3_3"] = {
        "3_3_0": {"pcc_p": pcc1, "pcc_n": pcc2},
        "3_3_1": {"pcc_m": pcc_m, "motion_ok": motion_ok},
        "3_3_2": {"pcc_t": pcc_t, "temp_ok": temp_ok},
        "3_3_3": {"delay": delay, "alarm_ok": alarm_ok},
        "3_3_4": {"music_idx": music_idx}
    }
    print(f"[green]Done.[/green]")
    return data_dict

def save_to_pickle(data_dict):
    """
    Save collected solutions to a pickle file.
    """
    with open("answer_sheet.pickle", "wb") as f:
        pickle.dump(data_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check the solution and zip the files.')
    parser.add_argument("--uid", type=str, help="Your University ID. e.g. 1234567")
    parser.add_argument("--task", type=int, choices=[1, 2, 3], help="The task number to check (1, 2, or 3).")
    args = parser.parse_args()

    # Run tests for the specified task
    if args.task:
        success = run_pytest_tests(args.task)
        if not success:
            console.print("\n[bold yellow]⚠️  Fix the test failures.[/bold yellow]")
            sys.exit(1)

    # Zip files if UID is provided
    if args.uid:
        # Run all tests
        data_dict = {}
        res = run_pytest_tests_all()
        
        if res:
            console.print("\n[bold green]✅ All tests passed![/bold green]")
            data_dict = collect_solution_1(data_dict)
            data_dict = collect_solution_2(data_dict)
            data_dict = collect_solution_3(data_dict)
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