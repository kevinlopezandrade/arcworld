import copy
import importlib
import json
import os
import random
import shutil
import zipfile
from inspect import getsource
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from matplotlib.colors import ListedColormap, Normalize

# from arc_constants import *
from arcworld.dsl.arc_constants import *

# from arc_types import *
from arcworld.dsl.arc_types import *

# from dsl import *
from arcworld.dsl.functional import *

String = str
Example = Dict[str, Grid]
Task = List[Example]
Tasks = Dict[str, Task]
TasksSplit = Dict[str, Tasks]


colormapping = {
    0: "ZERO",
    1: "ONE",
    2: "TWO",
    3: "THREE",
    4: "FOUR",
    5: "FIVE",
    6: "SIX",
    7: "SEVEN",
    8: "EIGHT",
    9: "NINE",
}

color_name_mapping = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "pink",
    7: "orange",
    8: "light blue",
    9: "dark red",
}


def format_grid(grid: List[List[int]]) -> Grid:
    """converts from list of lists to tuple of tuples"""
    return tuple(tuple(row) for row in grid)


def get_data(path: String) -> TasksSplit:
    """loads tasks"""
    data = {}
    file_names = os.listdir(path)
    num_files = len(file_names)
    for file_name in tqdm.tqdm(file_names, total=num_files):
        task_path = os.path.join(path, file_name)
        task_name = file_name.rstrip(".json")
        with open(task_path) as fp:
            task = json.load(fp)
        data[task_name] = task
    data_formatted = {"train": dict(), "test": dict()}
    for task_name, task in data.items():
        for group in ["train", "test"]:
            task_formatted = []
            for example in task[group]:
                example_formatted = {
                    "input": format_grid(example["input"]),
                    "output": format_grid(example["output"]),
                }
                task_formatted.append(example_formatted)
            data_formatted[group][task_name] = task_formatted
    return data_formatted


def plot_task(task: Task) -> None:
    """plots a task"""
    cmap = ListedColormap(
        [
            "#000",
            "#0074D9",
            "#FF4136",
            "#2ECC40",
            "#FFDC00",
            "#AAAAAA",
            "#F012BE",
            "#FF851B",
            "#7FDBFF",
            "#870C25",
        ]
    )
    norm = Normalize(vmin=0, vmax=9)
    args = {"cmap": cmap, "norm": norm}
    height = 2
    width = len(task)
    figure_size = (width * 4, height * 4)
    figure, axes = plt.subplots(height, width, figsize=figure_size)
    if width == 1:
        axes[0].imshow(task[0]["input"], **args)
        axes[1].imshow(task[0]["output"], **args)
        axes[0].axis("off")
        axes[1].axis("off")
    else:
        for column, example in enumerate(task):
            axes[0, column].imshow(example["input"], **args)
            axes[1, column].imshow(example["output"], **args)
            axes[0, column].axis("off")
            axes[1, column].axis("off")
    figure.set_facecolor("#1E1E1E")
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


def demo_tasks(tasks: TasksSplit, num_tasks: Integer) -> None:
    """plots a set of tasks"""
    options = tuple(tasks["train"].keys())
    for i in range(num_tasks):
        taskname = random.choice(options)
        task = tasks["train"][taskname]
        print(taskname)
        plot_task(task)


def generate_key(length: Integer) -> String:
    """generates a random task identifier"""
    key = ""
    for i in range(length):
        if random.choice((True, False)):
            key += chr(random.randint(48, 57))
        else:
            key += chr(random.randint(97, 122))
    return key


def read_base_programs(
    path: String, test_run: Boolean
) -> Tuple[List[Tuple[String, String]], List[Tuple[String, String]]]:
    """reads programs from json files"""
    with open(path) as fp:
        programs = json.load(fp)
    if test_run:
        programs = programs[:7]
    return programs


def flatten(
    name: String,
    program: String,
    argument_name: String = "anything",
    argument_type: String = "Any",
    return_type: String = "Any",
) -> String:
    """flattens a function string"""
    header = f"def {name}(\n    {argument_name}: {argument_type}\n) -> {return_type}:"
    lines = [header]
    tokens = [",", " ", "(", ")"]
    while program.count("(") > 1:
        for i, character in enumerate(program):
            if character == ")":
                attending, j = None, i
                while attending != "(":
                    j -= 1
                    attending = program[j]
                j -= 1
                attending = program[j]
                while attending not in tokens:
                    j -= 1
                    attending = program[j]
                call = program[j + 1 : i + 1]
                variable_name = f"x{len(lines)}"
                program = program.replace(call, variable_name)
                line = f"    {variable_name} = {call}"
                lines.append(line)
                break
    return_name = f"x{len(lines)}"
    lines.append(f"    {return_name} = {program}")
    lines.append(f"    return {return_name}({argument_name})")
    program = "\n".join(lines)
    return program


def squeeze(program):
    program = program.split("\n")[1:-2]
    squeezed = program[-1].split(" = ")[1]
    table = {
        line.split(" = ")[0].lstrip(): line.split(" = ")[1] for line in program[:-1]
    }
    tokens = ["(", ")", ",", " "]
    while True:
        table_new = dict()
        halt = True
        for key in table.keys():
            for start in tokens:
                for end in tokens:
                    subprogram = f"{start}{key}{end}"
                    if subprogram in squeezed:
                        squeezed = squeezed.replace(
                            subprogram, f"{start}{table[key]}{end}"
                        )
                        halt = False
        if halt:
            break
    return squeezed


def get_unrolled(
    definitions: List[Tuple[String, String]]
) -> List[Tuple[String, Callable]]:
    """converts program strings to functions"""
    header = "from arcworld.dsl.functional import *\n"
    header += "from arcworld.dsl.arc_types import *\n"
    header += "from arcworld.dsl.arc_constants import *\n\n\n"
    flattened = [flatten(name, program) for name, program in definitions]
    flattened = "\n\n\n".join(flattened)
    programs = header + flattened
    with open("tempfile.py", "w") as f:
        f.write(programs)
    module = importlib.import_module("tempfile")
    module = importlib.reload(module)
    unrolled = [(name, getattr(module, name)) for name, _ in definitions]
    os.remove("tempfile.py")
    return unrolled


def save_tasks(tasks: Tasks, path: String, key_length: Integer) -> Dict[String, String]:
    """saves a set of tasks"""
    if os.path.exists(path):
        shutil.rmtree(path)
        while os.path.isdir(path):
            pass
    os.makedirs(path)
    tasks_path = os.path.join(path, "tasks")
    os.makedirs(tasks_path)
    task_names = set()
    task_name_mapping = dict()
    tasks_path = os.path.join(path, "tasks")
    pbar = tqdm.tqdm(tasks.items(), desc="saving tasks")
    for key, task in pbar:
        task_name = generate_key(key_length)
        while task_name in task_names:
            task_name = generate_key(key_length)
        task_names.add(task_name)
        task_name_mapping[task_name] = key
        task_filename = f"{task_name}.json"
        task_path = os.path.join(tasks_path, task_filename)
        split = {"train": task[:-1], "test": [task[-1]]}
        with open(task_path, "w") as fp:
            json.dump(split, fp)
    return task_name_mapping


def test_solvers(tasks: TasksSplit, path: String) -> None:
    """checks whether the created programs solve the created tasks"""
    solvers_module_path = f"{path}.solvers"
    solvers_module = importlib.import_module(solvers_module_path)
    solvers_module = importlib.reload(solvers_module)
    total = 0
    successful = 0
    iterable = tasks["train"].items()
    pbar = tqdm.tqdm(iterable, desc="testing programs (0/0)")
    n = len(tasks["train"]) - 1
    for i, (task_name, train_examples) in enumerate(pbar):
        total += 1
        try:
            solver_name = f"solve_{task_name}"
            solver = getattr(solvers_module, solver_name)
            test_examples = tasks["test"][task_name]
            examples = train_examples + test_examples
            for example in examples:
                predicted = solver(example["input"])
                assert predicted == example["output"]
            successful += 1
        except:
            print(task_name)  # pass
        ratio = f"{successful}/{total}"
        if i == n:
            desc = f"{ratio} tasks solved"
        else:
            desc = f"testing programs ({ratio})"
        pbar.set_description(desc)


def save_task_name_mapping(
    path: String, taskname_mapping: Dict[String, String]
) -> None:
    """saves the task label to task specifier mapping"""
    mapping_path = os.path.join(path, "mapping.json")
    formatted = []
    for task_name, task_specifier in taskname_mapping.items():
        pair = f'    "{task_name}": "{task_specifier}"'
        formatted.append(pair)
    formatted = ",\n".join(formatted)
    formatted = f"{{\n{formatted}\n}}\n"
    with open(mapping_path, "w") as fp:
        fp.write(formatted)


def generate_shapes(
    min_obj_pixels: Integer,
    max_obj_pixels_proxy: Integer,
    num_objs_per_size_proxy: Integer,
    augmentations_selection: List[String],
) -> IndicesSet:
    """generates random shapes"""
    augm_scheme = lambda f: compose(normalize, fork(combine, identity, f))
    augmentations_options = [
        ("Identity", normalize),
        ("AddVerticallyMirrored", augm_scheme(vmirror)),
        ("AddHorizontallyMirrored", augm_scheme(hmirror)),
        (
            "AddHVMirrored",
            augm_scheme(fork(combine, augm_scheme(hmirror), augm_scheme(vmirror))),
        ),
        ("AddDiagonallyMirrored", augm_scheme(dmirror)),
        ("AddCounterdiagonallyMirrored", augm_scheme(cmirror)),
        (
            "AddDCMirrored",
            augm_scheme(fork(combine, augm_scheme(dmirror), augm_scheme(cmirror))),
        ),
        (
            "AddMirrored",
            augm_scheme(
                fork(
                    combine,
                    fork(combine, augm_scheme(hmirror), augm_scheme(vmirror)),
                    fork(combine, augm_scheme(dmirror), augm_scheme(cmirror)),
                )
            ),
        ),
        ("AddBox", augm_scheme(box)),
        ("AddOutBox", augm_scheme(outbox)),
        ("AddDiagonalLine", augm_scheme(fork(connect, ulcorner, lrcorner))),
        ("AddCounterdiagonalLine", augm_scheme(fork(connect, llcorner, urcorner))),
        (
            "AddCross",
            augm_scheme(
                fork(
                    combine,
                    fork(connect, ulcorner, lrcorner),
                    fork(connect, llcorner, urcorner),
                )
            ),
        ),
    ]
    augmentations = []
    for name, program in augmentations_options:
        if name in augmentations_selection:
            augmentations.append((name, program))
    random_shapes = set()
    pixel_range = range(min_obj_pixels, max_obj_pixels_proxy + 1)

    # What does the below line do ?
    end_index = max_obj_pixels_proxy - min_obj_pixels
    pbar = tqdm.tqdm(pixel_range, desc="generating random shapes (0)")
    neighboring_schemes = [neighbors, dneighbors]
    for i, n in enumerate(pbar):
        for k in range(num_objs_per_size_proxy):
            for neighboring_scheme in neighboring_schemes:
                shap = {(0, 0)}
                for i in range(
                    k - 1
                ):  # Number of times you are adding a random pixel to the initial pixel.
                    neighborhood = mapply(neighboring_scheme, shap)
                    candidates = difference(neighborhood, shap)
                    pixel = random.choice(tuple(candidates))
                    shap.add(pixel)
                shap = normalize(frozenset(shap))
                for name, augmentation in augmentations:
                    augmented = augmentation(shap)
                    random_shapes.add(augmented)
        if i == end_index:
            desc = f"generated {len(random_shapes)} random shapes"
            pbar.set_description(desc)
        else:
            desc = f"generating random shapes ({len(random_shapes)})"
            pbar.set_description(desc)
    return frozenset(random_shapes)


def get_proto_objs(shapes: IndicesSet, max_obj_dimension: Integer) -> Objects:
    """turns shapes into proto-objects"""
    bounding_function = lbind(greater, max_obj_dimension)
    small_enough = compose(bounding_function, compose(decrement, height))
    narrow_enough = compose(bounding_function, compose(decrement, width))
    filter_function = fork(both, small_enough, narrow_enough)
    recoloring_function = lbind(recolor, FIVE)
    shapes_filtered = sfilter(shapes, filter_function)
    proto_objs = apply(recoloring_function, shapes_filtered)
    return proto_objs


def get_locations(grid_shape, max_obj_dimension, margin, speedup_factor):
    h, w = grid_shape
    d = max_obj_dimension + 4 * margin
    locations = set()
    for i in range(0, h - d + 1, speedup_factor):
        for j in range(0, w - d + 1, speedup_factor):
            locations.add((i, j))
    return locations


def sample_object(
    grid_shape: Grid,
    max_obj_dimension: Integer,
    margin: Integer,
    selection: Tuple[Object],
    palette_scheme: Tuple[Integer],
    locations: Indices,
    occupied: Indices,
) -> Tuple[Object, Indices]:
    """chooses a random object and placement"""
    h, w = grid_shape
    d = max_obj_dimension + 4 * margin
    for i, j in occupied:
        locations_pruned = set()
        for a, b in locations:
            if i < a or i >= a + d or j < b or j >= b + d:
                locations_pruned.add((a, b))
        locations = locations_pruned
    if len(locations) == 0:
        return None, locations
    shift_vector = double(astuple(margin, margin))
    shift_function = rbind(add, shift_vector)
    locations_shifted = apply(shift_function, locations)
    loc_base = random.choice(tuple(locations_shifted))
    obj = random.choice(selection)
    offset_i = random.randint(0, max_obj_dimension - height(obj))
    offset_j = random.randint(0, max_obj_dimension - width(obj))
    offset = (offset_i, offset_j)
    location = add(loc_base, offset)
    obj = shift(obj, location)
    obj_color = random.choice(palette_scheme)
    obj = recolor(obj_color, obj)
    return obj, locations


def interactive(path: String, description_generator: Callable) -> None:
    """interactive demo of random tasks"""
    fname = path.rstrip(".zip")
    archive = zipfile.ZipFile(path, "r")
    with archive.open(os.path.join(fname, "mapping.json"), "r") as fp:
        mapping = json.load(fp)
    keys = tuple(mapping.keys())
    while True:
        key = random.choice(keys)
        with archive.open(os.path.join(fname, "tasks", f"{key}.json"), "r") as fp:
            task = json.load(fp)
        plot_task(task["train"])
        user_input = input('type enter to reveal task, "e" to exit demo')
        text_description = description_generator(mapping[key])
        print(text_description)
        if user_input == "e":
            break


def format_solver(obfuscated: String, program: String) -> String:
    """postprocess program string"""
    solver_name = f"solve_{obfuscated}"
    flattened = flatten(solver_name, program, "I", "Grid", "Grid")
    lines = flattened.split("\n")
    body = "\n".join(lines[:-2])
    main_call = lines[-2].split(" = ")[1]
    main_call = f"\n    O = {main_call}"
    return_statement = "\n    return O"
    program = body + main_call + return_statement
    return program


def save_solvers(
    solver_getter: Callable,
    task_names: List[String],
    programs_lists: Dict[String, List[Tuple[String, String]]],
    kmapper: Dict[String, String],
    path: String,
) -> None:
    """saves solvers programs"""
    program_mappings = {
        group_name: {name: program for name, program in programs_list}
        for group_name, programs_list, in programs_lists.items()
    }
    programs = "from arcworld.dsl.functional import *\n"
    programs += "from arcworld.dsl.arc_constants import *\n\n\n"
    pbar = tqdm.tqdm(task_names, desc="saving programs")
    for task_name in pbar:
        solver = solver_getter(
            **program_mappings, task_name=task_name, obfuscated=kmapper[task_name]
        )
        programs += f"\n{solver}\n\n"
    solvers_path = os.path.join(path, "solvers.py")
    with open(solvers_path, "w") as fp:
        fp.write(programs)


def train_test_split(
    tasks: Tasks, key_mapper: Dict[String, String], num_test_examples: Integer
) -> TasksSplit:
    """train test split"""
    tasks_split = {"train": dict(), "test": dict()}
    for name, task in tasks.items():
        key = key_mapper[name]
        tasks_split["train"][key] = task[:-num_test_examples]
        tasks_split["test"][key] = task[-num_test_examples:]
    return tasks_split


def verify(grid):
    if not isinstance(grid, tuple):
        return False
    if not 30 >= len(grid) > 0:
        return False
    if not isinstance(grid[0], tuple):
        return False
    if not 30 >= len(grid[0]) > 0:
        return False
    if not len(set([len(r) for r in grid])) == 1:
        return False
    if not all([all([0 <= x <= 9 for x in r]) for r in grid]):
        return False
    return True


def read_solver_programs(path, module, n=None, cutoff=None):
    function_names = []
    with open(path, "r") as fp:
        for line in fp.readlines():
            if line.startswith("def "):
                function_name = line.split(" ")[1].split("(")[0]
                function_names.append(function_name)
    programs = dict()
    if n is not None:
        function_names = function_names[:n]
    if cutoff is not None:
        function_names = function_names[cutoff:]
    for function_name in function_names:
        key = function_name.split("_")[1]
        programs[key] = getattr(module, function_name)
    return programs


def cross_augmentator(
    disallow_duplicate_outputs: bool,
    disallow_identities: bool,
    disallow_unicolored: bool,
    exclude_original: bool,
    n: int,
    cutoff: int,
    save_to: str = None,
):
    import eval_solvers
    import solvers

    train_data = get_data("../data/training")
    train_solvers = read_solver_programs("solvers.py", solvers, n, cutoff)
    working = set()
    new_tasks = {}
    for task_key, task in tqdm.tqdm(train_data["train"].items()):
        for program_key, program in train_solvers.items():
            if exclude_original and task_key == program_key:
                continue
            broken = False
            output_grids = set()
            new_task = {"train": [], "test": []}
            n_train = len(task)
            for i, ex in enumerate(task + train_data["test"][task_key]):
                try:
                    hat = program(ex["input"])
                    assert verify(hat)
                    if disallow_identities:
                        assert hat != ex["input"]
                    if disallow_unicolored:
                        assert numcolors(hat) > 1
                    if disallow_duplicate_outputs:
                        assert hat not in output_grids
                        output_grids.add(hat)
                except:
                    broken = True
                    break
                if i < n_train:
                    new_task["train"].append({"input": ex["input"], "output": hat})
                else:
                    new_task["test"].append({"input": ex["input"], "output": hat})
            if not broken:
                new_tasks[task_key + program_key] = new_task
                working.add((task_key, program_key))
    found = len(working)
    considered = len(train_data["train"]) * len(train_solvers)
    print(f"found {found}/{considered} ({found/considered*100:.2f}%)")

    if save_to is not None:
        if os.path.exists(save_to):
            shutil.rmtree(save_to)
            while os.path.isdir(save_to):
                pass
        os.makedirs(save_to)
        pbar = tqdm.tqdm(new_tasks.items(), desc="saving tasks")
        for key, task in pbar:
            task_path = os.path.join(save_to, f"{key}.json")
            with open(task_path, "w") as fp:
                json.dump(task, fp)
    return new_tasks
