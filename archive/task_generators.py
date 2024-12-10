import random
import simpy
from scheduler import Task

def cpu_bound_task_generator(env, scheduler, load_level="low"):
    task_id = 0

    if load_level == "low":
        arrival_interval = 5  
        duration_range = (3, 8)  
    elif load_level == "moderate":
        arrival_interval = 3 
        duration_range = (5, 15)  
    elif load_level == "high":
        arrival_interval = 1  
        duration_range = (10, 20)  
    else:
        raise ValueError(f"Unknown load_level: {load_level}")

    while True:
        task_id += 1
        duration = random.randint(*duration_range)
        priority = random.randint(1, 5)
        task = Task(f"CPU-Task-{task_id}", priority, duration)
        env.process(scheduler.schedule_task(task))
        yield env.timeout(random.expovariate(1 / arrival_interval))
