# File and Directory structure

## Directory structure:

### StepMania song directory structure:
Each song is a folder containing a `.sm` file and an audio file. Each song
exists within a song pack directory, i.e.:
`{song_pack_name}/{song_name}/`.

### Raw data folders:
`data/{raw_dataset_name}`

This folder should have the same structure as the StepMania folders.

### Train/test datasets:
`datasets/{dataset_name}`

Contains files generated from `bin/create_dataset.py`.

### Generated data:
`gen/{dataset_name}`

Should contain the same structure as the StepMania folders, but with a formatted wav file.
Typically is copied over from data using `bin/copy_dataset_songs.py`.

### Models:
`models/{model_name}`

Saved parameter weights for the trained models.

### Threhsolds:
`threshold/{threshold_name}`

The learned thresholds for the model. Generated from bin/get_thresholds.py.


## Scripts:
Located in the `bin` folder. Some conventions about the code: Parameters that end in `name` often refer to datasets/folders under a subdirectory, e.g., `datasets` as specified above. Parameters that end in `path` are classic file paths.

`preprocess_files.py`:
Modifies the raw data files to be suitable for training and testing/playing.
Converts .mp3/.ogg files into single channel, 44.1k sample rate .wav files.
Points the .sm file to read from the .wav file.

`create_dataset.py`:
Generates and saves Step Placement datasets from raw data, and save
train/test datasets to the .h5 format.

`convert_to_gen_dataset.py`:
Converts datasets for the Step Placement models into ones for the Step Generation models.

`train_test_split_dataset.py`:
Splits a dataset into train/test portions.

`train_step_placement_model.py`:
Trains the step placement model.

`train_step_generation_model.py`:
Trains the step generation model.

`get_thresholds.py`:
Computes the threshold dictionary for a step placement model on a given dataset. Typically uses the test set of the same dataset.

`generate_sm.py`:
Generates a .sm file from a processed .wav file.

`process_dataset.py`:
Applies `generate_sm.py` across a folder of songs. Each song should be a
StepMania song directory structure, with each folder containing a .wav file.
If the songs already contain `.sm` files (from either a previous run or from the original dataset), then use `--infer_bpm` to copy the BPM to speed up runtime.
This is run from inside the song folder.
Ex:
```
# Current directory: deepStep/gen/songfolder
python ../../bin/process_dataset.py \
	../../models/fraxtil-placement.py \
	../../models/fraxtil-thresholds.json
```

`evalute_step_placement.py`:
Takes the parameter weights from an existing Step Placement model and runs several performance metrics on them.

`evaluate_step_generation.py`:
Takes the parameter weights from an existing Step Generation model and runs several performance metrics on them.

`evaluate_bpm_estimation.py`:
Takes a generated song directory, and compares it to the BPM of the original dataset.

`export_songs.py`:
Prepares generated files for export, reduces file size.
Should operate on a copy of the generated files.
Removes images, other audio files, and other files in general.

## Source files
These exist in the deepSM file.

`SMData.py`:
Contains the `SMFile` class, which contains the information from a source `.sm` file.

`beat_time_converter.py`:
Contains utilities for converting between beat-aligned, frame-aligned, and time-aligned formats.

`SMDataset.py`:
Contains a PyTorch Dataset for Step Placement training.
These contain for each frame:
* FFT features
* Difficulty of each note set
* Frame-aligned step position labels
* Frame-aligned step type labels (optional)
These are loaded from file, or directly created from `SMDUtils.py`.

`SMGenDataset.py`:
Contains a PyTorch Dataset for Step Generation training.
The timesteps for this dataset is at each note time, which is specified from
	`SMDataset`'s step position labels.
For each timestep, these contain:
* FFT features
* Beats before
* Beats after
* Difficulty

These are loaded from file, or directly created from `SMDUtils.py`.

`SMDUtils.py`:
A utility file for manipulating `SMDataset` and `SMGenDataset` objects.
Contains functions to:
* Read datasets from file
* Create `SMDataset`s from raw files
* Split datasets into train/test components
* Save `SMDataset`s

`convert_to_gen_dataset.py`:
Converts Step Placement datasets into Step Generation datasets.

`wavutils.py`:
Utilities for manipulating wav files.
Appropriately pads the wav file to allow for context windows in the FFT at the front and back of the audio file.
Also computes Mel-scale spectrograms.

`NNModel.py`:
A base class for neural networks, inheriting from PyTorch's nn.Module.
Includes extra functions to abstract loading data, training, and prediction.

`StepPlacement.py`:
The class for the StepPlacement model. Inherits from NNModel.

`StepGeneration.py`:
The class for the StepGeneration model. Inherits from NNModel.

`post_processing.py`:
Functions to process the outputs of the models for more playable stepmaps.
Contains functions to:
* optimize get_thresholds
* Smooth output preditions
* Format and process outputs from the Step Generation model into actual predictions, and fills in blank steps.
* Filtering of triple notes.
* Removing notes that are too close in time.

`bpm_estimator.py`:
Estimates the BPM given the frame-aligned positions of the steps, extracts the fundamental BPM from source `.sm` files, and selects the optimal BPM from a list of candidates.
Note that BPM optimization is fairly unoptimized.

`beat_alignment.py`:
Given a BPM, identify the best offset for the notes to minimize the sum of subdivisions.
Additionally, return the beat-aligned notes to the learned subdivisions.

`generate_sm_file.py`:
Given the beat-aligned steps and other `.sm` file features, generate a final `.sm` file.
