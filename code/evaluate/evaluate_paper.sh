
nohup python -u evaluate_time.py TransE random digg_nodelocation_TransE.pkl digg_rellocation_TransE.pkl> TransE_100 &

nohup python -u evaluate_time.py CDK random digg_nodelocation_CDK.pkl > CDK_100 &

nohup python -u evaluate_time.py DPGE random digg_nodelocation_DPGE.pkl digg_rellocation_DPGE.pkl digg_nodevariance_DPGE.pkl > DPGE_100 &

nohup python -u evaluate_time.py KG2EEL random digg_nodelocation_kg2e_el.pkl digg_rellocation_kg2e_el.pkl digg_nodevariance_kg2e_el.pkl digg_relvariance_kg2e_el.pkl > kg2e_100 &
nohup python -u baseline_evaluate.py > baseline