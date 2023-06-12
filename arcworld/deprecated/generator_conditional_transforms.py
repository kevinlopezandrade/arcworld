from arcworld.deprecated.generator_utils import *


def sample_input_grid(
    satisfying_objects: Shapes,
    unsatisfying_objects: Shapes,
    num_objects_range: Coordinate,
    background_color_options: Tuple[Integer],
    max_obj_dimension: Integer,
    grid_dimensions_range: Coordinate,
    margin: Integer,
    speedup_factor: Integer,
    palette_scheme: Tuple[Integer],
) -> Tuple[Grid, Shapes, Integer]:
    """samples an input grid"""
    background_color = random.choice(background_color_options)
    min_grid_dim, max_grid_dim = grid_dimensions_range
    h = random.randint(min_grid_dim, max_grid_dim)
    w = random.randint(min_grid_dim, max_grid_dim)
    grid_shape = (h, w)
    grid = canvas(background_color, grid_shape)
    min_num_objs, max_num_objs = num_objects_range
    num_objs = random.randint(min_num_objs, max_num_objs)
    num_satisfying_objs = random.randint(1, num_objs - 1)
    num_unsatisfying_objs = num_objs - num_satisfying_objs
    num_objs_mapper = {
        "satisfying": num_satisfying_objs,
        "unsatisfying": num_unsatisfying_objs,
    }
    objs_mapper = {
        "satisfying": tuple(satisfying_objects),
        "unsatisfying": tuple(unsatisfying_objects),
    }
    selected_objs = {"satisfying": set(), "unsatisfying": set()}
    locations = get_locations(grid_shape, max_obj_dimension, margin, speedup_factor)
    occupied = frozenset({})
    fill_order = ["satisfying", "unsatisfying"][:: random.choice([-1, 1])]
    for group in fill_order:
        while len(selected_objs[group]) < num_objs_mapper[group]:
            obj, locations = sample_object(
                grid_shape=grid_shape,
                max_obj_dimension=max_obj_dimension,
                margin=margin,
                selection=objs_mapper[group],
                palette_scheme=palette_scheme,
                locations=locations,
                occupied=occupied,
            )
            # random_shape = random.choice(objs_mapper[group])

            # obj, locations = sample_object_test(
            #     grid_shape=grid_shape,
            #     max_obj_dimension=max_obj_dimension,
            #     margin=margin,
            #     shape=random_shape,
            #     palette_scheme=palette_scheme,
            #     locations=locations,
            #     occupied=occupied,
            # )

            # testing_grid.place_object(random_shape)

            if obj is None:
                return None, frozenset({}), background_color
            occupied = backdrop(obj)
            selected_objs[group].add(obj)
            grid = paint(grid, obj)
    selected_objs = frozenset(
        selected_objs["satisfying"] | selected_objs["unsatisfying"]
    )
    return grid, selected_objs, background_color


def construct_output_grid(
    background_color: Integer,
    input_grid: Grid,
    selection: Shapes,
    condition: Callable,
    transformation: Callable,
    apply_to_original: Callable,
    apply_to_transformed: Callable,
    apply_to_unsat: Callable,
) -> Grid:
    """constructs the output grid from the sampled input grid and transformation"""
    output_grid = canvas(background_color, shape(input_grid))
    originals = sfilter(selection, condition)
    original = merge(originals)
    transformed = mapply(transformation, originals)
    kept = merge(selection - originals)
    output_grid = paint(output_grid, apply_to_unsat(kept))
    output_grid = paint(output_grid, apply_to_original(original))
    output_grid = paint(output_grid, apply_to_transformed(transformed))
    return output_grid


def sample_task(
    objs: Shapes,
    conditions: List[Tuple[String, Callable]],
    transformations: List[Tuple[String, Callable]],
    palette_schemes: List[Tuple[String, Tuple[Integer]]],
    meta_schemes: List[Tuple[String, String]],
    unsat_schemes: List[Tuple[String, Boolean]],
    parameters: Dict,
    conditions_map: Dict[String, Shapes],
    transformations_map: Dict[String, Shapes],
) -> Tuple[Task, String]:
    """samples a task"""
    condition_name, condition = random.choice(conditions)
    transformation_name, transformation = random.choice(transformations)
    meta_scheme_name, meta_scheme = random.choice(meta_schemes)
    unsat_name, unsat_scheme = random.choice(unsat_schemes)
    collapse = lambda obj: frozenset({})
    if meta_scheme == "simple_transform":
        filter_function = matcher(compose(size, last), ONE)
        palette_schemes_subset = sfilter(palette_schemes, filter_function)
        palette_scheme_name, palette_scheme = random.choice(palette_schemes_subset)
        color_int = palette_scheme[0]
        apply_to_original = collapse
        apply_to_transformed = identity
    else:
        filter_function = matcher(compose(size, last), TWO)
        palette_schemes_subset = sfilter(palette_schemes, filter_function)
        palette_scheme_name, palette_scheme_base = random.choice(palette_schemes_subset)
        palette_scheme, outcolor = palette_scheme_base[:1], palette_scheme_base[1]
        color_int = outcolor
        apply_to_transformed = lbind(recolor, outcolor)
        if meta_scheme == "simple_recolor":
            apply_to_original = collapse
        elif meta_scheme == "overpaint":
            apply_to_original = identity
    if unsat_scheme:
        apply_to_unsat = collapse
    else:
        apply_to_unsat = identity

    # This is always the same, since we only pass one thing to sample.
    # There it can be avoided to check, and do it before entering.
    satisfying_objects = (
        conditions_map[condition_name] & transformations_map[transformation_name]
    )
    unsatisfying_objects = objs - conditions_map[condition_name]
    task = []
    num_failed_attempts = 0
    min_num_examples, max_num_examples = parameters.pop("num_examples_range")
    num_examples = random.randint(min_num_examples, max_num_examples)
    max_failed_attempts = parameters.pop("max_failed_attempts")
    min_obj_diversity = parameters.pop("min_obj_diversity")
    has_enough_satisfying_objs = len(satisfying_objects) >= min_obj_diversity
    has_enough_unsatisfying_objs = len(unsatisfying_objects) >= min_obj_diversity
    full_selections = []
    if has_enough_satisfying_objs and has_enough_unsatisfying_objs:
        while len(task) < num_examples and num_failed_attempts <= max_failed_attempts:
            input_grid, selection, background_color = sample_input_grid(
                satisfying_objects=satisfying_objects,
                unsatisfying_objects=unsatisfying_objects,
                palette_scheme=palette_scheme,
                **parameters,
            )
            if input_grid is None:
                num_failed_attempts += 1
                continue
            output_grid = construct_output_grid(
                background_color=background_color,
                input_grid=input_grid,
                selection=selection,
                condition=condition,
                transformation=transformation,
                apply_to_original=apply_to_original,
                apply_to_transformed=apply_to_transformed,
                apply_to_unsat=apply_to_unsat,
            )
            example = {"input": input_grid, "output": output_grid}
            task.append(example)
            full_selections.append(selection)
    too_few_examples = len(task) < min_num_examples
    has_identities = False
    for example in task:
        if example["input"] == example["output"]:
            has_identities = True
            break
    if too_few_examples or has_identities:
        task = None
    task_name = f"{condition_name}_{transformation_name}_{meta_scheme_name}_{color_int}_{unsat_name}"
    return task, task_name, full_selections


def get_solver(
    conditions_mapping: Dict[String, String],
    transformations_mapping: Dict[String, String],
    task_name: String,
    obfuscated: String,
    objects_program: String = "objects(I, T, T, T)",
) -> String:
    """constructs the solver function based on the task name"""
    # import ipdb
    # ipdb.set_trace()
    (
        condition_name,
        transformation_name,
        meta_scheme_name,
        color,
        unsatisfying_scheme_name,
    ) = task_name.split("_")
    condition_program = conditions_mapping[condition_name]
    transformation_program = transformations_mapping[transformation_name]
    color_string = colormapping[int(color)]
    satisfying_objects_program = f"sfilter({objects_program}, {condition_program})"
    transformed_objects_program = (
        f"mapply({transformation_program}, {satisfying_objects_program})"
    )
    if meta_scheme_name in ["SR", "OP"]:
        transformed_objects_program = (
            f"recolor({color_string}, {transformed_objects_program})"
        )
    if meta_scheme_name in ["SR", "ST"]:
        program = f"cover(I, merge({satisfying_objects_program}))"
    else:
        program = "I"
    if unsatisfying_scheme_name == "T":
        covered = f"merge(difference({objects_program}, {satisfying_objects_program}))"
        program = f"cover({program}, {covered})"
    program = f"paint({program}, {transformed_objects_program})"
    program = format_solver(obfuscated, program)
    return program


def get_cond_ht(proto_objs, conditions):
    ordered = order(proto_objs, identity)
    hashes = apply(hash, ordered)
    condnames = apply(first, conditions)
    df = pd.DataFrame(index=hashes, columns=condnames)
    for h, o in tqdm.tqdm(pair(hashes, ordered), desc="constructing conditions table"):
        for cn, cf in conditions:
            df.loc[h, cn] = cf(o)
    return df.astype(int)


def get_trans_ht(proto_objs, transformations):
    ordered = order(proto_objs, identity)
    hashes = apply(hash, ordered)
    transnames = apply(first, transformations)
    df = pd.DataFrame(index=hashes, columns=transnames)
    for h, o in tqdm.tqdm(
        pair(hashes, ordered), desc="constructing transformations table"
    ):
        for tn, tf in transformations:
            df.loc[h, tn] = hash(tf(o))
    return df


def check_well_definedness(
    full_selections, condition, transformation, cond_ht, trans_ht
):
    cname, condfun = condition
    tname, tfun = transformation
    full_selection = totuple(merge(full_selections[:-1]))
    full_selection = apply(normalize, full_selection)
    full_selection = apply(lbind(recolor, FIVE), full_selection)
    full_sat_hashes = sorted(apply(hash, full_selection))
    test_selection = totuple(full_selections[-1])
    test_selection = apply(normalize, test_selection)
    test_selection = apply(lbind(recolor, FIVE), test_selection)
    test_sat_hashes = sorted(apply(hash, test_selection))
    c_sat_subs = cond_ht.loc[full_sat_hashes, :]
    c_target = c_sat_subs[cname]
    c_src = c_sat_subs.drop(columns=[cname])
    overdetermined_conds = []
    for col in c_src.columns:
        if all(c_target == c_src[col]):
            overdetermined_conds.append(col)
    if len(overdetermined_conds) > 0:
        c_sat_subs_test = cond_ht.loc[test_sat_hashes, :]
        c_target_test = c_sat_subs_test[cname]
        for col in overdetermined_conds:
            if any(c_target_test != c_sat_subs_test[col]):
                return False
    t_sat_subs = trans_ht.loc[full_sat_hashes, :]
    t_target = t_sat_subs[tname]
    t_src = t_sat_subs.drop(columns=[tname])
    overdetermined_trans = []
    for col in t_src.columns:
        if all(t_target == t_src[col]):
            overdetermined_trans.append(col)
    if len(overdetermined_trans) > 0:
        t_sat_subs_test = trans_ht.loc[test_sat_hashes, :]
        t_target_test = t_sat_subs_test[tname]
        for col in overdetermined_trans:
            if any(t_target_test != t_sat_subs_test[col]):
                return False
    return True


def mix(
    programs_list: List[Tuple[String, String]], mixes: List[String]
) -> List[Tuple[String, String]]:
    """constructs programs of depth 2"""
    ## For this to work, I cannot use objects it has to be text of the DSL.
    ## It can actually be. If I do something like Transformation.to_dsl()
    programs = []
    for outer_name, outer_program in tqdm.tqdm(programs_list, desc="mixing programs"):
        for inner_name, inner_program in programs_list:
            if outer_name != inner_name:
                for mixer in mixes:
                    name = f"{inner_name}{mixer.capitalize()}{outer_name}"
                    program = f"fork({mixer}, {outer_program}, {inner_program})"
                    programs.append((name, program))
    return programs


def filter_programs_by_size(
    programs_list: List[Tuple[String, String]], max_len: Integer
) -> List[Tuple[String, String]]:
    """filters programs by size"""
    return [
        (n, p) for n, p in programs_list if len(flatten(n, p).split(":")[-1]) <= max_len
    ]


def generate_tasks(
    shapes_config: Dict,
    conditions_path: String,
    transformations_path: String,
    palette_schemes: List[Tuple[String, Tuple[Integer]]],
    meta_schemes: List[Tuple[String, String]],
    unsat_schemes: List[Tuple[String, Boolean]],
    parameters: Dict,
    path: String,
    save_progs: Boolean,
    test_progs: Boolean,
    test_run: Boolean,
    condition_mixes: Boolean,
    transformation_mixes: Boolean,
    max_len: Integer,
    seed: Integer,
    key_length: Integer,
) -> TasksSplit:
    """generates a set of tasks"""
    random.seed(seed)
    conditions_list = read_base_programs(conditions_path, test_run)
    transformations_list = read_base_programs(transformations_path, test_run)
    if len(condition_mixes) > 0:
        conditions_list = mix(conditions_list, condition_mixes)
    if len(transformation_mixes) > 0:
        transformations_list = mix(transformations_list, transformation_mixes)
    if max_len is not None:
        conditions_list = filter_programs_by_size(conditions_list, max_len)
        transformations_list = filter_programs_by_size(transformations_list, max_len)
    conditions = get_unrolled(conditions_list)
    transformations = get_unrolled(transformations_list)

    # I avoid the use of TempFiles in pyhton.

    ## Generate the Shapes.
    shapes = generate_shapes(**shapes_config)
    proto_objs = get_proto_objs(shapes, parameters["max_obj_dimension"])

    ### Conditions Processing.
    conditions_map = dict()
    for cn, cf in tqdm.tqdm(conditions, desc="constructing conditions map"):
        conditions_map[cn] = sfilter(proto_objs, cf)

    ## Get Valid Transformations Objects Candidates.
    transformations_map = dict()
    for name, transformation in tqdm.tqdm(
        transformations, desc="constructing transformations map"
    ):
        not_empty = chain(positive, size, transformation)
        not_identity = compose(flip, fork(equality, identity, transformation))

        # Outbox is the box surrounding the shape, it does not contain any coordinate
        # of the shape. So if after applying the transformation, the shape intersects
        # with the outbox then I'm out of bounds. And the transformation produced
        # is not valid. With Margin = 1 I ensure that no transformation produces
        # a shape that touches my outbox.
        not_out_of_bounds = matcher(
            compose(
                size,
                fork(
                    difference,
                    compose(toindices, transformation),
                    compose(backdrop, power(outbox, parameters["margin"])),
                ),
            ),
            ZERO,
        )
        not_empty_or_identity = fork(both, not_empty, not_identity)
        is_valid = fork(both, not_empty_or_identity, not_out_of_bounds)
        transformations_map[name] = sfilter(proto_objs, is_valid)
    # cond_ht = get_cond_ht(proto_objs, conditions)
    # trans_ht = get_trans_ht(proto_objs, transformations)

    ### Task sampling.
    considered = 0
    attempted = 0
    succeeded = 0
    block_size = len(meta_schemes) * len(unsat_schemes)
    end_index = len(conditions) - 1
    tasks = {}
    pbar = tqdm.tqdm(conditions, desc="generating tasks (0/0/0)")
    get_desc = lambda: f"generating tasks ({succeeded}/{attempted}/{considered})"
    update_pbar = lambda: pbar.set_description(get_desc())
    for i, condition in enumerate(pbar):
        condition_name = condition[0]
        condition_candidates = conditions_map[condition_name]
        for transformation in transformations:
            transformation_name = transformation[0]
            transformation_candidates = transformations_map[transformation_name]
            candidates = condition_candidates & transformation_candidates

            ### Up to there just choosing the valid transformation candidates and the shapes with the right conditions.

            if len(candidates) >= parameters["min_obj_diversity"]:
                for meta_scheme in meta_schemes:
                    for unsat_scheme in unsat_schemes:
                        considered += 1
                        attempted += 1
                        task, task_name, full_selections = sample_task(
                            objs=proto_objs,
                            conditions=[condition],
                            transformations=[transformation],
                            palette_schemes=palette_schemes,
                            meta_schemes=[meta_scheme],
                            unsat_schemes=[unsat_scheme],
                            parameters=copy.deepcopy(parameters),
                            conditions_map=conditions_map,
                            transformations_map=transformations_map,
                        )
                        well_defined = True
                        # check_well_definedness(
                        #    full_selections=full_selections,
                        #    condition=condition,
                        #    transformation=transformation,
                        #    cond_ht=cond_ht,
                        #    trans_ht=trans_ht
                        # )
                        if task is not None and well_defined:
                            succeeded += 1
                            tasks[task_name] = task
                        update_pbar()
            else:
                considered += block_size
                update_pbar()
        if i == end_index:
            desc = f"generated {succeeded} tasks (att: {attempted}, cons. {considered})"
            pbar.set_description(desc)
    taskname_mapping = save_tasks(tasks, path, key_length)
    key_mapper = {key: name for name, key in taskname_mapping.items()}
    save_task_name_mapping(path, taskname_mapping)
    tasks_split = train_test_split(tasks, key_mapper, 1)
    if save_progs:
        task_names = list(tasks.keys())
        save_solvers(
            get_solver,
            task_names,
            {
                "conditions_mapping": conditions_list,
                "transformations_mapping": transformations_list,
            },
            key_mapper,
            path,
        )
        if test_progs:
            test_solvers(tasks_split, path)
    return tasks_split


def generate_text_description(task_name: str) -> str:
    """generates a task description"""
    (
        condition_name,
        transformation_name,
        metascheme_name,
        color,
        removal,
    ) = task_name.split("_")
    color_name = color_name_mapping[int(color)]
    text_description = (
        f'Apply transformation "{transformation_name}" to '
        f'all objects which satisfy condition "{condition_name}". '
    )
    if metascheme_name in ["SR", "OP"]:
        text_description += f'Color all transformed objects with color "{color_name}". '
    if metascheme_name == "OP":
        text_description += (
            "Paint the transformed objects on top of the unchanged original objects. "
        )
    if removal == "T":
        text_description += "Remove all objects which do not satisfy the condition. "
    return text_description
