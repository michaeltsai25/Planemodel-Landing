from PlaneEnv.env.FlightEnv import PlaneEnvironment
from PlaneEnv.utils import gridsearch_tensorforce
from PlaneEnv.env.AnimatePlane import animate_plane
from tensorforce import Agent
from PlaneEnv.utils import runner
from PlaneEnv.env.graph_utils import plot_reward
from PlaneEnv.env.graph_utils import plot_flight
from pathlib import Path
import os
from PlaneEnv.env.AnimatePlane import read_txt

def main(animate=False):
    # Instantiane our environment
    environment = PlaneEnvironment()
    # Instantiate a Tensorforce agent
    agent = Agent.create(agent="ppo",environment=environment, batch_size=100)
    param_grid_list = {}
    param_grid_list["PPO"] = {
        "batch_size": [10],
        "update_frequency": [10],
        "learning_rate": [1e-3],
        "subsampling_fraction": [0.3],
        "optimization_steps": [100],
        "likelihood_ratio_clipping": [0.1],
        "discount": [1.0],
        "estimate_terminal": [False],
        "multi_step": [30],
        "learning_rate_critic": [1e-3],
        "exploration": [0.01],
        "variable_noise": [0.000],
        "l2_regularization": [0.1],
        "entropy_regularization": [0.01],
        "network": ["auto"],
        "size": [32],
        "depth": [4],
    }

    num=6

    rewards = gridsearch_tensorforce(
        environment,
        param_grid_list["PPO"],
        max_step_per_episode=1000,
        n_episodes= 25000,
        num=num
    )

    plot_reward(rewards, num)
    
    
    # Animate last run positions
    if animate:
        animate_plane()

cur_path = os.path.dirname(__file__)

def test():
    Path(os.path.join(cur_path, "env", f"Exp_{1}", str(1))).mkdir(parents=True, exist_ok=True)
    path = os.path.join(cur_path, cur_path, "env", f"Exp_{1}", str(1), "test.txt")
    with open(path, 'w') as txt_file:
        txt_file.write("test")

def test_2():
    plot_reward([1, 2, 3, 4], 5)

def flight_plot():
    #with open("PlaneEnv/positions.txt") as pos:
    """
    pos_l = list(pos.read())
    dis = pos_l[0]
    hei = pos_l[1]
    plot_flight(dis, hei, 6)
    """
        
    pos_vec, theta_vec = read_txt()
    dis = pos_vec[0]
    hei = pos_vec[1]
    plot_flight(dis, hei, 7)

if __name__ == "__main__":
    # main(animate=True)
    #animate_plane()
    # test_2()
    flight_plot()

