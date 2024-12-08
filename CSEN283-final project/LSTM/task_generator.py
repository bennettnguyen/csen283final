import random
import simpy
from scheduler import Task

def task_generator(env, scheduler,arrival_interval=3):
    task_id = 0
    while True:
        task_id += 1
        duration = random.randint(3, 10)
        priority = random.randint(1, 5)
        task = Task(f"Task-{task_id}", priority, duration)
        env.process(scheduler.schedule_task(task))
        yield env.timeout(random.expovariate(1 / arrival_interval))  
