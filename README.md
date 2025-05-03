# CS240-Examining-Machine-Unlearning-on-Image-Data
CS 240: Artificial Intelligence and Machine Learning Lab Project on Examining Machine Unlearning on Image Data by Sameer Arind Patil(23B1035) and Harsh Suthar(23B1067)
To run this repo, follow the below steps
1. clone this repo to your desktop
2. Install all the dependencies by running the following command
```bash
pip3 install requirements.txt
```
3. Run the analysis.ipynb notebook, this will plot the confusion matrices for each of the 6 models, while also geenrating the .pth files for the models, these files are not provided in this repo directly due to file size limit of 25MB on github(yep, the models are indeed big!)
4. Put the generated models into ./models folder, this is the path set for the Streamit pathname
5. Download the tar file for the CIFAR-10 dataset from <href>https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz</href>, put it in your cloned repository, and run the following command on the terminal to unzip it.
```bash
tar -xvzf cifar-10-python.tar.gz
```
6. Run the following command on the terminal
```bash
python3 -m streamlit run inference_comparision.py
```
7. This will redirect you to the webpage, where you can see the result plots, while also test the outputs from various models on different test input images 
