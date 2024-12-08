import simpy

TIME_QUANTUM = 5
HIGH_LOAD_THRESHOLD = 70

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

    def schedule_task(self, task):
        with self.cpu.request(priority=task.priority) as req:  
            yield req  
            if self.algorithm == "Round Robin":
                yield self.env.timeout(min(task.duration, TIME_QUANTUM))
                task.duration -= TIME_QUANTUM
                if task.duration > 0:
                    self.env.process(self.schedule_task(task))
            elif self.algorithm == "Priority Scheduling":
                yield self.env.timeout(task.duration)

    def update_algorithm(self, predicted_load):
        if predicted_load > HIGH_LOAD_THRESHOLD and self.algorithm != "Priority Scheduling":
            self.algorithm = "Priority Scheduling"
            self.switch_log.append((self.env.now, "Priority Scheduling"))
        elif predicted_load <= HIGH_LOAD_THRESHOLD and self.algorithm != "Round Robin":
            self.algorithm = "Round Robin"
            self.switch_log.append((self.env.now, "Round Robin"))
