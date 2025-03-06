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

import doctest
import argparse
import os
from datetime import datetime
import zipfile
import sys
import pickle
import subprocess
import importlib.util

LAB_ID = 4
TASK_ID = [1, 2, 3]
DATA_ROOT = "./data/"

SOL = False  # DO NOT CHANGE THIS
suffix = "_sol" if SOL else ""

if SOL:
    # Import solution modules
    from task_4_1_sol import task_4_1
    from task_4_2_sol import task_4_2
    from task_4_3_sol import task_4_3
else:
    # Import student modules
    from task_4_1 import task_4_1
    from task_4_2 import task_4_2
    from task_4_3 import task_4_3

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
        f"task_4_1{suffix}.py",
        f"task_4_2{suffix}.py",
        f"task_4_3{suffix}.py",
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

def collect_doctest_results(task_file):
    """
    Run doctests for a specific task file as a continuous unit and return the results.
    
    Parameters
    ----------
    task_file : str
        The name of the task file to test (e.g., 'task_4_1.py')
    
    Returns
    -------
    dict : Contains test results with passed/failed counts and failure details
    """
    print(f"Collecting doctest results for {task_file} ...", end="\t")
    
    # Load the module dynamically
    try:
        spec = importlib.util.spec_from_file_location(task_file[:-3], task_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"[red]Failed to load {task_file}: {str(e)}[/red]")
        return {
            "tests_run": 0,
            "tests_failed": 0,
            "passed": False,
            "failures": [f"ModuleLoadError: {str(e)}"]
        }
    import numpy as np
    # Set up globals for doctest
    globs = {
        "np": np,
        "DATA_ROOT": DATA_ROOT
    }
    # Add module attributes to globals
    globs.update({name: getattr(module, name) for name in dir(module) if not name.startswith('__')})
    
    # Run doctests with the module
    test_result = doctest.testmod(
        module,
        globs=globs,
        verbose=False,
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    )
    
    result_dict = {
        "tests_run": test_result.attempted,
        "tests_failed": test_result.failed,
        "passed": test_result.failed == 0,
        "failures": []
    }
    
    # If there were failures, run again with verbose to capture details
    if test_result.failed > 0:
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            doctest.testmod(
                module,
                globs=globs,
                verbose=True,
                optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
            )
        result_dict["failures"] = f.getvalue().splitlines()
    
    print(f"[green]Done.[/green]")
    return result_dict

def collect_solution_1(data_dict):
    """
    Collect doctest results for Task 4-1.
    """
    task_file = f"task_4_1{suffix}.py"
    data_dict["task_4_1"] = collect_doctest_results(task_file)
    return data_dict

def collect_solution_2(data_dict):
    """
    Collect doctest results for Task 4-2.
    """
    task_file = f"task_4_2{suffix}.py"
    data_dict["task_4_2"] = collect_doctest_results(task_file)
    return data_dict

def collect_solution_3(data_dict):
    """
    Collect doctest results for Task 4-3.
    """
    task_file = f"task_4_3{suffix}.py"
    data_dict["task_4_3"] = collect_doctest_results(task_file)
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