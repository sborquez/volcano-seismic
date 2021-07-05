# Volcano-Seismic Classifier

Automatic classification of seismic signals from Llaima volcano (Chile).

## Quickstart

```bash
# Copy the repository
project_dir="$(pwd)/volcano-seismic_classifier"
git clone https://github.com/sborquez/volcano-seismic_classifier.git $project_dir
# Run DataScience docker and mount project folder.
sudo docker run --name volcano-seismic_classifier --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 -v "$project_dir:/rapids/host/notebooks sborquez/datascience:latest-gpu"
# Download Datasets
(rapis)$ python /rapids/host/notebooks get_data.py --all
# Open your browser http://localhost:8888/lab? and go to host/notebooks
```

## Bibliography
* [Classification of seismic signals at Villarrica volcano (Chile) using neural networks and genetic algorithms](https://www.sciencedirect.com/science/article/abs/pii/S0377027308006355)
* [Pattern recognition applied to seismic signals of the Llaima volcano (Chile): An analysis of the events' features](http://repositorio.uchile.cl/handle/2250/126669)
* [Attention is All You Need](https://arxiv.org/abs/1706.03762v5)
* [Machine Learning for Volcano-Seismic Signals: Challenges and Perspectives MACHINE LEARNING FOR VOLCANO-SEISMIC SIGNALS: CHALLENGES AND PERSPECTIVES 1 Machine Learning for Volcano-seismic Signals: Challenges and Perspectives](https://ieeexplore.ieee.org/document/8310698)
* [Using CNN To Classify Spectrograms of Seismic Events From Llaima Volcano (Chile)](https://ieeexplore.ieee.org/document/8489285)
* [Llaima Dataset. In-depth comparison of deep artificial neural network architectures on seismic events classification](https://www.sciencedirect.com/science/article/pii/S2352340920305217)
* [Discriminating seismic events of the Llaima volcano (Chile) based on spectrogram cross-correlations](https://www.sciencedirect.com/science/article/abs/pii/S0377027318301616)
* [In-depth comparison of deep artificial neural network architectures on seismic events classification](https://www.sciencedirect.com/science/article/abs/pii/S0377027319306171)
* [AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778)

## Documentation

* [Tensorflow](https://www.tensorflow.org/)
