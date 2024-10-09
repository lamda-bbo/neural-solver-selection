# Neural Solver Selection for Combinatorial Optimization
Official implementation of Arxiv preprint paper: "".

**Authors**: Chengrui Gao, Haopu Shang, Ke Xue, Chao Qian. 

**Abstract**: Machine learning has increasingly been employed to solve NP-hard combinatorial optimization problems, resulting in the emergence of neural solvers that demonstrate remarkable performance, even with minimal domain-specific knowledge. To date, the community has created numerous open-source neural solvers with distinct motivations and inductive biases. While considerable efforts are devoted to designing powerful single solvers, our findings reveal that existing solvers typically demonstrate complementary performance across different problem instances. This suggests that significant improvements could be achieved through effective coordination of neural solvers at the instance level.
In this work, we propose the first general framework to coordinate the neural solvers, which involves feature extraction, selection model, and selection strategy, aiming to allocate each instance to the most suitable solvers. To instantiate, we collect several typical neural solvers with state-of-the-art performance as alternatives, and explore various methods for each component of the framework. We evaluated our framework on two extensively studied combinatorial optimization problems, Traveling Salesman Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP). Experimental results show that the proposed framework can effectively distribute instances and the resulting composite solver can achieve significantly better performance (e.g., reduce the optimality gap by 0.88\% on TSPLIB and 0.71\% on CVRPLIB) than the best individual neural solver with little extra time cost.


## Usage
We provide scripts for both training and testing, named `run_experiment.sh` and `run_test.sh`. For instance, you can train a selection model on TSP by:

```python
python run.py --config_name config_TSP.yml --loss rank --seed 2024 --gpu_id 0
```
Evaluations of pretrained checkpoints can be simply conducted through:
```python
python run.py --gpu_id 0 
--load config_TSP.yml_rank_2024 # name of pretrained checkpoint
--test_file TSPLIB  # test dataset
--exp_name config_TSP.yml_rank_TSPLIB   # name of experimental results
```
The pretrained checkpoints of ranking model on TSP and CVRP are provided in the `train_logs/` folder. 

Most hyperparameters are configured using `.yml` files, such as `config_CVRP.yml` and `config_TSP.yml`, where default hyperparameters to reprofuce our results are provided. 

You can use `experiments/generate_experiments_shell.py` and `experiments/generate_test_shell.py` to generate more scripts for training and testing selection models. 

## Reference
If you found this reposity is useful for your research, please cite the paper:

