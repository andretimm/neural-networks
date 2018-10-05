import numpy as np
from scipy.stats import bernoulli, sem
from math import sqrt, log
import matplotlib.pyplot as plt

number_of_iterations = 100
time_horizon = 10000

arm1_actual_mean = [0.9,0.9,0.55]
arm2_actual_mean = [0.6,0.8,0.45]

ALPHA = 4


def update_mean(old_mean,current_reward,count):
	new_mean = old_mean + float(current_reward-old_mean)/count
	return new_mean

def ucb1(x1,x2):
	global regret_ucb, number_of_times_optimal_arm_played_ucb
	number_of_times_arms_played = [1,1]
	arm_means = [0.0,0.0]
	regret = 0.0
	arm_ucbs = [0.0,0.0]
	temp = [0]*time_horizon
	for t in range(time_horizon):
		arm_ucbs[0] = arm_means[0] + sqrt((float(ALPHA)*log(t+1)) / (2*number_of_times_arms_played[0])) 
		arm_ucbs[1] = arm_means[1] + sqrt((float(ALPHA)*log(t+1)) / (2*number_of_times_arms_played[1])) 

		arm_played = arm_ucbs.index(max(arm_ucbs)) 

		if arm_played == optimal_arm:
			number_of_times_arms_played[0] += 1
			temp[t] += 1
			number_of_times_optimal_arm_played_ucb[t] += 1
			arm_means[0] = update_mean(arm_means[0],x1[t],number_of_times_arms_played[0])

		else:
			number_of_times_arms_played[1] += 1
			arm_means[1] = update_mean(arm_means[1],x2[t],number_of_times_arms_played[1])
			regret += x1[t] - x2[t]

		regret_ucb[t] += regret

	optimal_arm_all_iterations.append(temp)

def plot(x_axis,y_axis, standard_error, title, y_label):
	plt.errorbar(x_axis,y_axis,standard_error)
	plt.suptitle("UCB: "+title, fontsize=16)
	plt.xlabel("Trials: ",fontsize = 14)
	plt.ylabel(y_label,fontsize = 14)
	plt.show()	

		

for j in range(len(arm1_actual_mean)):

	number_of_times_optimal_arm_played_ucb = [0]*time_horizon
	regret_ucb = [0.0]*time_horizon
	optimal_arm_all_iterations = []

	optimal_arm = np.argmax([arm1_actual_mean[j],arm2_actual_mean[j]])

	for i in range(number_of_iterations):
		# Bernoulli RV
		x1 = bernoulli.rvs(arm1_actual_mean[j],size=time_horizon)
		x2 = bernoulli.rvs(arm2_actual_mean[j],size=time_horizon)

		ucb1(x1,x2)

		print "Round ",i

	problem = "Problem "+str(arm1_actual_mean[j])+","+str(arm2_actual_mean[j])
	plot(range(time_horizon),[float(r)/100 for r in regret_ucb],float('NaN'),problem,"Regret")
	plot(range(time_horizon),[float(n)/100 for n in number_of_times_optimal_arm_played_ucb],sem(optimal_arm_all_iterations), problem,"Number of times optimal arm played")