# Diffusion Segmentation Project for ETH Deep Learning A23

The project currently adheres to the following structure:
- Python files (`.py`) live in the `code` directory
- Jupyter notebooks (`.ipynb`) live in the `notebooks` directory
- Datasets are placed in the `data` directory, with the following structure:
    - `data`
        - `{dataset_name_1}`
            - `img`
                - `{image_file_name_1}.jpg`
                - `{image_file_name_2}.jpg`
                - ...
                - `{image_file_name_n}.jpg`
            - `gt`
                - `{ground_truth_segmentation_file_name_1}.png`
                - `{ground_truth_segmentation_file_name_2}.png`
                - ...
                - `{ground_truth_segmentation_file_name_n}.png`
        - `{dataset_name_2}`
            - ...
            