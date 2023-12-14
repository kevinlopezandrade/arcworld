from arcworld.training.models.utils import load_model_from_dir, predict
from arcworld.utils import decode_json_task

weights_path = "PATH_TO_YOUR_WEIGHTS_IN_HYDRA_OUTPUT_DIR"
model = load_model_from_dir(weights_path)

path_task = "PATH_TO_YOUR_TAKS_IN_JSON"
gt, pred = predict(model, decode_json_task(path_task))
