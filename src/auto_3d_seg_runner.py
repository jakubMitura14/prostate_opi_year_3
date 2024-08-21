import numpy as np
import os
import pandas as pd
import SimpleITK as sitk
from monai.apps.auto3dseg.bundle_gen import BundleGen
from monai.apps.auto3dseg.data_analyzer import DataAnalyzer
from monai.apps.auto3dseg.ensemble_builder import EnsembleRunner
from monai.apps.auto3dseg.hpo_gen import NNIGen
from monai.apps.auto3dseg.utils import export_bundle_algo_history, import_bundle_algo_history
from monai.apps.utils import get_logger
from monai.auto3dseg.utils import algo_to_pickle
from monai.bundle import ConfigParser
from monai.transforms import SaveImage
from monai.utils import AlgoKeys, has_option, look_up_option, optional_import
from monai.utils.misc import check_kwargs_exist_in_class_init, run_cmd


import os
import pandas as pd
import yaml
import json
from monai.apps.auto3dseg.auto_runner import AutoRunner

os.environ["CUDA_VISIBLE_DEVICES"] = "0" #TODO remove this line
# env CUDA_VISIBLE_DEVICES=0
# main_dir="/root/data/post_norm_min_max_bias_on"
main_dir="/workspaces/prostate_opi_year_3/data/dataset/post_norm_min_max_bias_on"
train_folds_path="/workspaces/prostate_opi_year_3/data/AI4AR_uczacy_foldy.csv"
test_ids_path="/workspaces/prostate_opi_year_3/data/AI4ARtestowy.csv"

folds_df=pd.read_csv(train_folds_path)
test_ids_df=pd.read_csv(test_ids_path,header=None)

test_ids = test_ids_df.iloc[:, 0].to_numpy()

folds_df.columns


# Load the data
folds_df = pd.read_csv(train_folds_path)
test_ids_df = pd.read_csv(test_ids_path, header=None)
test_ids = test_ids_df.iloc[:, 0].to_numpy()

# def create_empty_label_if_not_exists(image_path, label_path):
#     # Check if the label file already exists
#     if not os.path.exists(label_path):
#         # Read the image from image_path
#         image = sitk.ReadImage(image_path)
        
#         # Get the size and spacing of the original image
#         size = image.GetSize()
#         spacing = image.GetSpacing()
#         origin = image.GetOrigin()
#         direction = image.GetDirection()
        
#         # Create a new image with the same size, spacing, origin, and direction
#         # but with all pixel values set to zero and of type uint16
#         empty_label = sitk.Image(size, sitk.sitkUInt16)
#         empty_label.SetSpacing(spacing)
#         empty_label.SetOrigin(origin)
#         empty_label.SetDirection(direction)
        
#         # Save the new image to label_path
#         sitk.WriteImage(empty_label, label_path)
#         print(f"Created empty label image at {label_path}")
#     else:
#         # print(f"Label image already exists at {label_path}")
#         pass

# Prepare the data dictionary
data_dict = {
    "training": [],
    "testing": [],
    "modality": "MRI"
}
def modify_string(string):
    if len(string) >=3:
        return string
    elif len(string) == 2:
        return "0" + string
    else:
        return "00" + string

# Populate the training data
for _, row in folds_df.iterrows():
    patient_id = row['Patient ID']
    fold = row['fold']
    ppath_folder=modify_string(str(patient_id))
    image_path = os.path.join(main_dir, ppath_folder, f"{str(patient_id)}_adc.nii.gz")
    label_path = os.path.join(main_dir,ppath_folder, f"{str(patient_id)}_lesion_union_adc.nii.gz")
    if not os.path.exists(label_path):
        label_path = os.path.join(main_dir,ppath_folder, f"{str(patient_id)}_empty.nii.gz")
    data_dict["training"].append({
        "fold": int(fold)-1,
        "image": image_path,
        "label": label_path
    })

# Populate the testing data
for test_id in test_ids:
    image_path = os.path.join(main_dir, str(test_id), f"{str(test_id)}_adc.nii.gz")
    label_path = os.path.join(main_dir,ppath_folder, f"{str(patient_id)}_lesion_union_adc.nii.gz")
    if not os.path.exists(label_path):
        label_path = os.path.join(main_dir,ppath_folder, f"{str(patient_id)}_empty.nii.gz")
        
    data_dict["testing"].append({
        "image": image_path,
        "label": label_path

    })

# Save the data dictionary to a YAML file
datalist_path = "/workspaces/prostate_opi_year_3/data/input.json"
with open(datalist_path, 'w') as yaml_file:
    json.dump(data_dict, yaml_file)


# Define the work directory
work_dir = "/workspaces/prostate_opi_year_3/data/work_dir_b"
os.makedirs(work_dir, exist_ok=True)

runner = AutoRunner(
    work_dir=work_dir,
    input={
        "modality": "MRI",
        "datalist": datalist_path,
        "dataroot": work_dir,
    },
)
# Prepare the configuration dictionary
# config_dict = {
#     "datalist": input_yaml_path,
#     "dataroot": main_dir,
#     "modality": "MRI"
# }

# # Save the configuration dictionary to a YAML file
# config_yaml_path = os.path.join(work_dir, "config.json")
# with open(config_yaml_path, 'w') as yaml_file:
#     json.dump(config_dict, yaml_file)

# print(f"Config YAML file saved to {config_yaml_path}")

# Initialize and run the AutoRunner
# runner = AutoRunner(work_dir=work_dir, input=config_yaml_path)


# print(runner.data_src_cfg_name)

# ensemble_runner = EnsembleRunner(
#     data_src_cfg_name=runner.data_src_cfg_name,
#     work_dir=runner.work_dir,
#     num_fold=runner.num_fold,
#     ensemble_method_name=runner.ensemble_method_name,
#     mgpu=int(runner.device_setting["n_devices"]) > 1,
#     **runner.kwargs,  # for set_image_save_transform
#     **runner.pred_params,
# )  # for inference
# ensemble_runner.run(runner.device_setting)




# example_to_predict="/workspaces/prostate_opi_year_3/data/dataset/post_norm_min_max_bias_on/1218/1218_lesion_union_adc.nii.gz"
# output_path="/workspaces/prostate_opi_year_3/data/results/1218_lesion_inferred.nii.gz"


runner.run()

