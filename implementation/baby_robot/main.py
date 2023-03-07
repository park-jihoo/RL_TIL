import utils
import numpy as np
from matplotlib import pyplot as plt

methods = []
rewards = []
mean_rewards = []

def run_multiple_tests( tester, max_steps = 2000, show_socket_percentages = True ):
    number_of_tests = 100
    number_of_steps = max_steps
    maximum_total_reward = 3600

    experiment = utils.SocketExperiment(socket_tester   = tester,
                                number_of_tests = number_of_tests,
                                number_of_steps = number_of_steps,
                                maximum_total_reward = maximum_total_reward)
    experiment.run()

    print(f'Mean Reward per Time Step = {experiment.get_mean_total_reward():0.3f}')
    print(f'Optimal Socket Selected = {experiment.get_optimal_selected():0.3f}')    
    print(f'Average Number of Trials Per Run = {experiment.get_mean_time_steps():0.3f}')
    if show_socket_percentages:
        print(f'Socket Percentages = {experiment.get_socket_percentages()}')
        
    rewards.append(experiment.get_cumulative_reward_per_timestep())        
    mean_rewards.append(f"{experiment.get_mean_total_reward():0.3f}")
    
if __name__ == "__main__":
    print(utils.socket_means)
    print("PowerSocket")
    run_multiple_tests( utils.SocketTester( utils.PowerSocket ) )
    print("\nEpsilonGreedy")
    run_multiple_tests(utils.EpsilonGreedySocketTester( epsilon = 0.2 ) )
    print("\nOptimalSocket")
    run_multiple_tests( utils.SocketTester(utils.OptimisticPowerSocket, initial_estimate = 20. ))
    print("\nUCB")
    run_multiple_tests( utils.SocketTester( utils.UCBSocket, confidence_level = 0.6 ))
    print("\nGaussianTS")
    run_multiple_tests(utils.SocketTester(utils.GaussianThompsonSocket ))
    
    
    test_names = ['Greedy','Epsilon Greedy','Optimistic Greedy','UCB','Thompson Sampling']

    plt.figure(figsize=(10,8))
    plt.yticks(np.arange(0., 4200, 600))

    for test in range(len(rewards)):
        plt.plot(rewards[test], label = f'{test_names[test]}')
        
    plt.legend()    
    plt.title('Mean Total Reward vs Time', fontsize=15)
    plt.xlabel('Time Step')
    plt.ylabel('Mean Total Reward (seconds of charge)')
    plt.savefig('./MeanTotalReward.png')