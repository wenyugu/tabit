# tabit

## Notes
Use <https://github.com/andywiggins/tab-cnn> as starter code. This project contains common audio preprocessing pipelines and some evaluation metrics but it is written in Keras. 

## Dataset
<https://github.com/marl/GuitarSet>

Make sure to use the *audio_mono-mic* subset which uses monophonic recording from reference microphone
Place the generated spectral features under `data/spec_repr` 

## Train and Test
`python3 train.py`

## Acknowledgement
We thank [1] for providing the code for preprocessing and [2] for making the dataset available and providing helpful visualization code in `interpreter.py`.

## Reference
[1] Andrew Wiggins and Youngmoo Kim. 2019. Guitar Tablature Estimationwith a Convolutional Neural Network. InProceedings of the 20th Inter-national Society for Music Information Retrieval Conference. ISMIR, Delft,The Netherlands, 284–291.

[2] Qingyang Xi, Rachel M. Bittner, Johan Pauwels, Xuzhou Ye, and Juan P.Bello. 2019.GuitarSet.