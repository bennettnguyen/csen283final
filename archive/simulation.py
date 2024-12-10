import simpy
from task_generators import cpu_bound_task_generator
from scheduler import AdaptiveScheduler
import random
from scheduler import Task



def io_bound_task_generator(env, scheduler, arrival_interval=3):
    task_id = 0
    while True:
        task_id += 1
        duration = random.randint(1, 5) 
        priority = random.randint(1, 5)  
        task = Task(f"IO-Task-{task_id}", priority, duration)
        env.process(scheduler.schedule_task(task))
        yield env.timeout(random.expovariate(1 / arrival_interval))

def mixed_task_generator(env, scheduler, arrival_interval=3):
    task_id = 0
    while True:
        task_id += 1
        if random.random() > 0.5:
            duration = random.randint(10, 20) 
            task_type = "CPU"
        else:
            duration = random.randint(1, 5)  
            task_type = "IO"
        priority = random.randint(1, 5)
        task = Task(f"{task_type}-Task-{task_id}", priority, duration)
        env.process(scheduler.schedule_task(task))
        yield env.timeout(random.expovariate(1 / arrival_interval))

def monitor(env, scheduler, predictor, utilization):
    while True:
        cpu_utilization = (scheduler.cpu.capacity - scheduler.cpu.count) / scheduler.cpu.capacity * 100
        utilization.append((env.now, cpu_utilization))
        predictor.update(cpu_utilization)
        predicted_load = predictor.predict()
        scheduler.update_algorithm(predicted_load)
        yield env.timeout(1)

def run_simulation(predictor, sim_time=100):
    env = simpy.Environment()
    cpu = simpy.PriorityResource(env, capacity=4)
    scheduler = AdaptiveScheduler(env, cpu, predictor)

def run_simulation(predictor, sim_time=100, load_level="low"):
    env = simpy.Environment()
    cpu = simpy.PriorityResource(env, capacity=4)
    scheduler = AdaptiveScheduler(env, cpu, predictor)

    env.process(cpu_bound_task_generator(env, scheduler, load_level=load_level))

    utilization = []
    env.process(monitor(env, scheduler, predictor, utilization))
    env.run(until=sim_time)
    return utilization, scheduler.switch_log
