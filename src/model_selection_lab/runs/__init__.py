"""Running a task: every input of a task on every model, with each answer saved.

- task_drafter.py: turns a plain description into a task file, using a helper model.
- task_loader.py: reads and checks a task YAML file (the instruction and the test inputs).
- batch_runner.py: sends every input to every chosen model and saves each answer, with
  a cost estimate before the run, a spending cap, and reuse of saved answers.
- results_storage.py: saves each result as one line in a .jsonl file, and finds saved
  answers that can be reused.
"""
