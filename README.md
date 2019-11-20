Code to reproduce the results of the paper.

# Usage
- Download:
  - The images segmentation.
  - The codebook.
  - The survey definitions (i.e. the files that specify the traversals).
- Extract local features to disk.
- Retrieve query images.

# Setup
This code is tested on Ubuntu 16.04. The cpp part depends on OpenCV 3.2.0.
The python packages needed are specified in `requirements.txt`.

We also provide a Dockerfile although it may not be necessary.

## Manual
#### cpp dependencies
- *OpenCV 3.2.0*
  - Archive download link: [link](https://github.com/opencv/opencv/archive/3.2.0.zip)
  - Installation instructions: [link](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)

- *(Optional) Faiss*
Only if you want to generate the codebook. An valid alternative is K-Means
scikit-learn implementation.

#### python dependencies
Run `pip3 install -r requirements.txt` to install all packages. 
Warning: this may change the version of your packages.
Recommended: Install only the packages you are missing.

#### Build wasabi2 and experiments
```bash
mkdir build
cd build
cmake ..
make
```


## Docker 
Build and run the docker image:
```bash
cd docker
make root # build
nvidia-docker run --volume="/path/to/wasabi2":"/home/ws" -it  -u root wasabi2 bash # run
```

## Datasets
### Extended-CMU-Seasons
Download all the segmentation:
```bash
cd meta/cmu/seg
./get_data.sh -1 # for all slices (<1G)
./get_data.sh 22 # for slice 22 only
```

Download the surveys i.e. the file lists specifying the traversals:
```bash
cd meta/cmu
./get_surveys.sh
```

### Symphony
Download the segmentation and the surveys:
```bash
cd meta/cmu/symphony/
./get_data.sh
```

### Codebooks
```bash

```


# Reproduce the experiments
## Symphony
Extract the local feature to disk
```bash
./scripts/extract_features.sh symphony 0
```

Run the retrieval:
```bash

```

Plot the retrieval scores:
```bash

```


## Extended-CMU-Seasons
For clarity, this section explains how to run the experiments for the park slice 22. 
For the rest of the park slices, just replace 22 with {23,24,25}.

For the urban slices, replaces the park training slices {18,19,20,21} with the urban ones {2,3,4,5}. 
And the park evaluation slices {22,23,24,25} with the urban ones {6,7,8}.

Extract the local feature to disk
```bash
./scripts/extract_features.sh cmu_park 22 # for park slice22

./scripts/extract_features.sh cmu_park -1 # for all park retrievals
./scripts/extract_features.sh cmu_urban -1 # for all park retrievals
```

Run the retrieval:
```bash
./scripts/wasabi2.sh cmu park_22 # for park slice22

./scripts/wasabi2.sh cmu park # for all park retrievals
./scripts/wasabi2.sh cmu urban # for all city retrievals
```

Generate the plots
```bash

```