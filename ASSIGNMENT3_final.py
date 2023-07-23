
import random
import simpy
import numpy as np
from numpy.random import RandomState
from scipy import stats
from tabulate import tabulate
#---------------------------Making assumptions----------------------------------
RANDOM_SEED = 42

NUM_PRE_FACILITIES = 3  # Number of preparation facilities in hospital, 
NUM_OP_THEATERS = 1  # Number of operation theaters in hospital, 
NUM_RE_FACILITIES = 5  # Number of Recovery facilities in hospital, 

PRE_TIME = 40      # Minutes it takes to Prepare a patient
OP_TIME = 20      # Minutes it takes to operate a patient
RE_TIME = 40      # Minutes it takes to recovery a patient

T_INTER = 25       # patient arrive at every ~7 minutes

SIM_TIME = 1000    # Simulation time in 20 minutes, not a real world time.

confidence = 0.95
NUM_OF_RUNS = 10
death_rate = 25

np.random.seed(RANDOM_SEED)

#------------------------Create services--------------------------------------------------
class Preparation(object):
    def __init__(self, env, num_pre_facilities, pre_time):
        self.env = env
        self.machine = simpy.Resource(env, num_pre_facilities)
        self.pre_time = pre_time
        self.data = []

    def Prepare(self, patient):
        self.pre_time = max(1, np.random.normal(PRE_TIME, 4))
        yield self.env.timeout(self.pre_time)
        print(" %s has been prepared"% patient[0] )


class Operation(object):
    def __init__(self, env, num_op_facilities, op_time):
        self.env = env
        self.machine = simpy.Resource(env, num_op_facilities)
        self.op_time = op_time
        self.data = []
    def Operate(self, patient):
        randonnumber = max(1, np.random.normal(OP_TIME, 4))
        yield self.env.timeout(randonnumber)
        # p is the probability for surviving
        p = random.randint(0, 100)
        if(p < death_rate):
            print(" %s has passed away :( "%patient[0] )
            patient[1] = True
            
        else:
            print(" %s has been successfully Operated"% patient[0])


class Recovery(object):
    def __init__(self, env, num_re_facilities, re_time):
        self.env = env
        self.machine = simpy.Resource(env, num_re_facilities)
        self.re_time = re_time
    def Recover(self, patient):
        self.re_time = max(1, np.random.normal(RE_TIME, 4))
        yield self.env.timeout(self.re_time)
        print(" %s has been recovered"% patient[0])
#------------------------Create patient logic----------------------------------
# Patient should have 3 services.
def Patient(env, name, pre, op, re):
    print('%s arrives at the preparation queue at %.2f.' % (name[0], env.now))
    with pre.machine.request() as request:
        yield request
        pre.data.append((env.now, len(pre.machine.queue)))
        
        print('%s enters the preparation at %.2f.' % (name[0], env.now))
        prep_request_list.append(env.now)
        yield env.process(pre.Prepare(name))
        print('%s leaves the preparation at %.2f.' % (name[0], env.now))
        prep_release_list.append(env.now)
    #------------Start operation service
    print('%s arrives at the Operation queue at %.2f.' % (name[0], env.now))
    with op.machine.request() as request:
        yield request
        op.data.append((env.now, op.machine.count))
        print('%s enters the Operation at %.2f.' % (name[0], env.now))
        op_request_list.append(env.now)
        yield env.process(op.Operate(name))
        print('%s leaves the Operation at %.2f.' % (name[0], env.now))
        op_release_list.append(env.now)
        if op_release_list[-1] - op_request_list[-1] > op.op_time:
            blocked_time_list.append(op_release_list[-1] - op_request_list[-1]- op.op_time)

    #------------Start Recovery service
    if name[1] == False:

        print('%s enters the Recovery at %.2f.' % (name[0], env.now))
        yield env.process(re.Recover(name))
        print('%s leaves the Recovery at %.2f.' % (name[0], env.now))

def Arrival(env, t_inter,pre,op,re):
    # Create 4 initial Patient
    for i in range(4):
        # we assume the patient is alive
        env.process(Patient(env, ['Patient %d' % i,False] , pre,op,re))

    # Create more patient while the simulation is running
    while True:
        yield env.timeout(random.randint(t_inter - 2, t_inter + 2))
        i += 1
        env.process(Patient(env, ['Patient %d' % i,False], pre,op,re))


def calc_ICs():
    sigma_queue = np.std(np.array(list_of_average_queue_length),ddof=1)
    sigma_op_ut = np.std(np.array(list_of_op_utilizations),ddof=1)
    sigma_prep_ut = np.std(np.array(list_of_prep_utilizations),ddof=1)
    sigma_blocked = np.std(np.array(list_of_blocked_rates),ddof=1)
    mean_queue = np.array(list_of_average_queue_length).mean()
    mean_op_ut = np.array(list_of_op_utilizations).mean()
    mean_prep_ut = np.array(list_of_prep_utilizations).mean()
    mean_blocked = np.array(list_of_blocked_rates).mean()
    mean = np.array([mean_queue, mean_prep_ut, mean_op_ut, mean_blocked])
    sigma = np.array([sigma_queue, sigma_prep_ut, sigma_op_ut, sigma_blocked])
    dof  = NUM_OF_RUNS - 1
    t_crit = np.abs(stats.t.ppf((1-confidence)/2,dof))
    inf, sup = [], []
    for i in range(mean.size):
        inf.append(mean[i] - sigma[i]*t_crit/np.sqrt(NUM_OF_RUNS))
        sup.append(mean[i] + sigma[i]*t_crit/np.sqrt(NUM_OF_RUNS))
    inf = np.round(inf,2)
    sup = np.round(sup,2)
    queue = ["Queue",mean[0], sigma[0], inf[0], sup[0]]
    idle_prep = ["Idle Prep",mean[1], sigma[1], inf[1], sup[1]]
    ut_op = ["Util Op",mean[2], sigma[2], inf[2], sup[2]]
    blocked = ["Blocked Rate",mean[3], sigma[3], inf[3], sup[3]]       
    all_data = [
        ["monitoring","mean","sigma","inf","sup"],
        queue,
        idle_prep,
        ut_op,
        blocked]
    return all_data


list_of_prep_utilizations, list_of_op_utilizations, list_of_average_queue_length, list_of_blocked_rates = [], [], [], []
for run in range(NUM_OF_RUNS):
    prep_request_list, prep_release_list = [],[]
    op_request_list, op_release_list = [],[]
    utilization_prep_list, utilization_op_list = [], []
    blocked_time_list = []
# Create an environment and start the setup process
    env = simpy.Environment()
    pre = Preparation(env, NUM_PRE_FACILITIES, PRE_TIME)
    op = Operation(env, NUM_OP_THEATERS, OP_TIME)
    re = Recovery(env, NUM_RE_FACILITIES, RE_TIME)
    env.process(Arrival(env, T_INTER,pre,op,re))

# Execute!
    env.run(until=SIM_TIME)


    npdata = np.array(pre.data)
    average = npdata[:, 1].mean()
    print('Entrance average queue: ', average)
    list_of_average_queue_length.append(average)

    for i in range(len(prep_release_list)):
        utilization_prep_list.append(prep_release_list[i] - prep_request_list[i])

    prep_sum_idle = np.nansum(utilization_prep_list)
    utilization_prep = 100 - round(prep_sum_idle/(NUM_PRE_FACILITIES * SIM_TIME) * 100, 2)
    print("Idle Prep: ", utilization_prep)
    list_of_prep_utilizations.append(utilization_prep)

    for i in range(len(op_release_list)):
        utilization_op_list.append(op_release_list[i] - op_request_list[i])

    op_sum_idle = np.nansum(utilization_op_list)
    utilization_op = round(op_sum_idle/(NUM_OP_THEATERS * SIM_TIME) * 100, 2)
    print("Util. Op: ", utilization_op)
    list_of_op_utilizations.append(utilization_op)

    blocked_time_sum = np.nansum(blocked_time_list)
    blocked_rate = round(blocked_time_sum/op_sum_idle * 100, 2)
    print("Blocked rate: ", blocked_rate)
    list_of_blocked_rates.append(blocked_rate)
    
table = calc_ICs()

print('Number of Preparation Facilities: ', NUM_PRE_FACILITIES)
print('Number of Recovery Facilities: ', NUM_RE_FACILITIES)
print(tabulate(table, headers="firstrow",tablefmt="fancy_outline"))