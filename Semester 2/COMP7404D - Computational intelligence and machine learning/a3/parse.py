def read_grid_mdp_problem_p1(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    problem = {
        "seed": None,
        "noise": None,
        "livingReward": None,
        "grid": [],
        "policy": [],
    }

    current_section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("seed:"):
            problem["seed"] = int(line.split(":")[1].strip())
        elif line.startswith("noise:"):
            problem["noise"] = float(line.split(":")[1].strip())
        elif line.startswith("livingReward:"):
            problem["livingReward"] = float(line.split(":")[1].strip())
        elif line.startswith("grid:"):
            current_section = "grid"
        elif line.startswith("policy:"):
            current_section = "policy"
        else:
            if current_section == "grid":
                problem["grid"].append(line.split())
            elif current_section == "policy":
                problem["policy"].append(line.split())

    return problem


def read_grid_mdp_problem_p2(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    problem = {
        "discount": None,
        "noise": None,
        "livingReward": None,
        "iterations": None,
        "grid": [],
        "policy": [],
    }

    current_section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("discount:"):
            problem["discount"] = float(line.split(":")[1].strip())
        elif line.startswith("noise:"):
            problem["noise"] = float(line.split(":")[1].strip())
        elif line.startswith("livingReward:"):
            problem["livingReward"] = float(line.split(":")[1].strip())
        elif line.startswith("iterations:"):
            problem["iterations"] = int(line.split(":")[1].strip())
        elif line.startswith("grid:"):
            current_section = "grid"
        elif line.startswith("policy:"):
            current_section = "policy"
        else:
            if current_section == "grid":
                problem["grid"].append(line.split())
            elif current_section == "policy":
                problem["policy"].append(line.split())

    return problem


def read_grid_mdp_problem_p3(file_path):
    # Your p3 code here
    problem = ""
    return problem
