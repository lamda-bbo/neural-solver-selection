python run.py --gpu_id 1 --load config_TSP.yml_rank_2024 --test_file TSPtest --exp_name config_TSP.yml_rank_TSPtest &
python run.py --gpu_id 0 --load config_TSP.yml_rank_2024 --test_file TSPLIB --exp_name config_TSP.yml_rank_TSPLIB &
wait;
python run.py --gpu_id 1 --load config_CVRP.yml_rank_2024 --test_file CVRPtest --exp_name config_CVRP.yml_rank_CVRPtest &
python run.py --gpu_id 0 --load config_CVRP.yml_rank_2024 --test_file CVRPLIB --exp_name config_CVRP.yml_rank_CVRPLIB &
wait;
