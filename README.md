Code to reproduce the results of the paper.

# Usage
- Download the images.
- Segment the images.
- Extract local features to file.
- Download/Generate the codebook.
- Download the survey definitions.
- Retrieve query images.

# Setup
This code is tested on Ubuntu 16, with the following libraries installed.

## Dependencies
### cpp
- OpenCV 3.2.0
Archive download link: [link](https://github.com/opencv/opencv/archive/3.2.0.zip)
Installation instructions: [link](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)

- (Optional) Faiss
Only if you want to generate the codebook. An valid alternative is K-Means
scikit-learn implementation.

### python
See requirements.txt for the missing packages.
Run `pip3 install -r requirements.txt` to install all packages. Warning: this may change the version of your packages.

## wasabi2

```bash
cmake
make
```


# Run the paper experiments.

## CMU-Seasons
This section presents an example on how to run the experiments for the park slice 22. 
For the rest of the park slices, just replace 22 with {23,24,25}.

For the urban slices, replaces the park training slices {18,19,20,21} with the urban ones {2,3,4,5}. 
And the park evaluation slices {22,23,24,25} with the urban ones {6,7,8}.


Download the survey definitions
```bash
cd meta/cmu/
./get_surveys.sh
```

Download the segmentation for slice 22.
(Alternatively, you can download the images and run the segmentation with the
model from [this repo](https://github.com/maunzzz/cross-season-segmentation).)
```bash
cd meta/cmu/seg
./get_data.sh 22
```

Extract image local features to file. TODO: what size ?

```bash

```

Download the codebook. This command actually downloads all codebook (~TODO
size).
```bash

```


Cluster
```bash

```

## Symphony

# Datasets
## Extended-CMU-Seasons
The CMU-Seasons dataset is quite big so you may want to download one slice only. 

Else, if you want all the slices used in the paper, you can run this command
line. WARNING: the total size of the archives takes 129.3G.

## Symphony
Run this script to get the Symphony retrieval dataset:

