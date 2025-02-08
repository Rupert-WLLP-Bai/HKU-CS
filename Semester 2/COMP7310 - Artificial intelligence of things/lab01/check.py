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



LAB_ID = 1
TASK_ID = [1, 2, 3]

SOL = False # DO NOT CHANGE THIS
suffix = "_sol" if SOL else ""

if SOL:
    # Import task modules
    from task_1_1_sol import task_1_1
    from task_1_2_sol import task_1_2
    from task_1_3_sol import task_1_3
else:
    # Import task modules
    from task_1_1 import task_1_1
    from task_1_2 import task_1_2
    from task_1_3 import task_1_3

def run_pytest_tests(task):
    """
    Run pytest for the specified task.

    Parameters
    ----------
    task : int
        The task number (1, 2, or 3).
    """
    console.print(f"\n[bold cyan]********* Running Tests for Task {LAB_ID}-{task} *********[/bold cyan]")
    
    # Define the test file based on the task
    test_file = f"task_1_{task}{suffix}.py"
    
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
        test_files.append(f"task_1_{task}{suffix}.py")
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
    try:
        for task in TASK_ID:
            visualize_task(task)
    except Exception as e:
        console.print(f"\n[bold red]❌ Error during visualization: {str(e)}[/bold red]")
        flag = False
    return flag

def visualize_task(task):
    """
    Call the visualize function for the specified task.

    Parameters
    ----------
    task : int
        The task number (1, 2, or 3).
    """
    
    if task == 1:
        task = task_1_1(1000)
    elif task == 2:
        task = task_1_2(1000)
    elif task == 3:
        task = task_1_3(1000)
    else:
        console.print(f"[bold red]❌ Invalid task number: {task}[/bold red]")
    task.visualize()

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
    submit_files = [f"task_1_1{suffix}.py", f"task_1_2{suffix}.py", f"task_1_3{suffix}.py"]
    submit_folders = ["fig"]  # Folder containing figures
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check the solution and zip the files.')
    parser.add_argument("--uid", type=str, help="Your University ID. e.g. 1234567")
    parser.add_argument("--task", type=int, choices=[1, 2, 3], help="The task number to check (1, 2, or 3).")
    args = parser.parse_args()

    # Run tests and visualize for the specified task
    if args.task:
        success = run_pytest_tests(args.task)
        visualize_task(args.task)
        
        if not success:
            console.print("\n[bold yellow]⚠️  Fix the test failures.[/bold yellow]")
            sys.exit(1)

    # Zip files if UID is provided
    if args.uid:
        # run all tests
        res = run_pytest_tests_all()
        if not res:
            console.print("\n[bold yellow]⚠️  Fix the test failures before zipping.[/bold yellow]")
            sys.exit(1)
        else:
            console.print("\n[bold green]✅ All tests passed![/bold green]")
        
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