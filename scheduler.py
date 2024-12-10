import random
import simpy


class Task:
    def __init__(self, name, priority, duration):
        self.name = name
        self.priority = priority
        self.duration = duration

class AdaptiveScheduler:
    def __init__(self, env, cpu, predictor):
        self.env = env
        self.cpu = cpu
        self.predictor = predictor
        self.algorithm = "Round Robin"
        self.switch_log = []

    def update_algorithm(self, predicted_load):
        HIGH_LOAD_THRESHOLD = 80
        MEDIUM_LOAD_THRESHOLD = 50

        if predicted_load > HIGH_LOAD_THRESHOLD and self.algorithm != "Priority Scheduling":
            self.algorithm = "Priority Scheduling"
            self.switch_log.append((self.env.now, "Priority Scheduling"))
        elif MEDIUM_LOAD_THRESHOLD < predicted_load <= HIGH_LOAD_THRESHOLD and self.algorithm != "SJF":
            self.algorithm = "SJF"
            self.switch_log.append((self.env.now, "SJF"))
        elif predicted_load <= MEDIUM_LOAD_THRESHOLD and self.algorithm != "Round Robin":
            self.algorithm = "Round Robin"
            self.switch_log.append((self.env.now, "Round Robin"))

    def schedule_task(self, task):
        with self.cpu.request(priority=task.priority) as req:
            yield req
            if self.algorithm == "Round Robin":
                yield self.env.timeout(min(task.duration, 5))
                task.duration -= 5
                if task.duration > 0:
                    self.env.process(self.schedule_task(task))
            elif self.algorithm == "Priority Scheduling":
                yield self.env.timeout(task.duration)
            elif self.algorithm == "SJF":
                yield self.env.timeout(task.duration / 2)  

def task_generator(env, scheduler, arrival_interval=3):
    task_id = 0
    while task_id < 5:
        task_id += 1
        duration = random.randint(3, 10)
        priority = random.randint(1, 5)
        task = Task(f"Task-{task_id}", priority, duration)
        env.process(scheduler.schedule_task(task))
        yield env.timeout(random.expovariate(1 / arrival_interval))

env = simpy.Environment()
cpu = simpy.PriorityResource(env, capacity=1)
scheduler = AdaptiveScheduler(env, cpu,predictor=any)

env.process(task_generator(env, scheduler, arrival_interval=3))

env.run()

print("start", scheduler.switch_log)
