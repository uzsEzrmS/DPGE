# DPGE : Diffusion Prediciton through Gaussian Embedding
## INTRODUCTION
We present an information diffusion prediction model, which makes use of Gaussian embedding and the transmission time interval to predict who will be infected in a cascade eventually. We named the model ``Diffusion Prediction through Gaussian Embedding''(DPGE).
## USAGE
We provide the Linux test version of DPGE in digg datasets, which the experiments results is shown in our model.<br>
Environment:<br>
* Python 2.7.12
* scipy 0.19.0
* numpy 1.11.3
## INPUT
The example of input file is valid in data/cascade_digg. The input of consists of the cascades extracte from the dataset. Each line of the input file represents a cascde diffused in the network, which is specified as the format "source_node node1 node2 ...." (can be either separated by blank or tab). Here is an input example:<br>
315 693 1038<br>
206 828 1 324 465<br>
348 35 113 119 115 145 126 48 27 124 110 57 157<br>
## Learning
The training codes is in code/model/ folder.<br>
sh ./model_run.sh<br>
The learned embeddings of nodes and other parameters learned are saved as pkl format in result folder.<br>
## Evaluating
The evaluation part is in code/evaluate/ folder.<br>
* evaluate_paper.sh
* result_paper.sh
The final results are saved in code/evaluate/ folder with txt format.
