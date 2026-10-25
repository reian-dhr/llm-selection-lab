"""Settings: every value the tool assumes, in one place.

Nothing here has to be changed to use the tool: these are defaults. They are gathered in
one file so that anyone can see what a run assumes, and change it in one place.

What is NOT here:
- facts about a model (its ID, its prices): see models/model_registry.py,
- what the user wants (the task, a length limit, a format): that is written in the task.
"""

# --- Calling a model ---

# The longest answer a model may write, in tokens (about 750 words). An answer that needs
# more is cut off. Raise this for tasks with long answers.
MAX_ANSWER_TOKENS = 1000

# Extra room, in tokens, given to models that think before answering. Their thinking
# counts toward the limit, so without this their answers would be cut off. Thinking is
# billed as output, so this is also the most a call can add to the cost.
THINKING_EXTRA_TOKENS = 4000

# How many times one call is tried before giving up: 1 first try + 2 retries.
MAX_TRIES = 3

# Seconds to wait after the 1st and after the 2nd failed try.
RETRY_WAITS_S = (1, 2)

# When a provider says how long to wait before trying again (after a rate limit), the
# wait follows it, up to this many seconds. Longer waits are not worth it: the call fails.
MAX_RETRY_WAIT_S = 60

# --- Turning a description into a task ---

# The helper model that reads a task description and writes the task file. It comes
# from a family outside the comparison, so its test inputs cannot favour a candidate.
TASK_DRAFTER_MODEL = "qwen-3-8-27b"

# How many test inputs it writes for a new task.
DRAFTED_INPUTS = 3

# Room for the task file it writes, in tokens. Groq's free tier refuses a request when it
# expects more than 1,000 tokens of answer a minute; 1,500 here passes that check, so the
# drafting prompt asks for compact task files.
DRAFTER_MAX_TOKENS = 1500

# The prompt that tells it how to do this. A private file, kept out of the code.
TASK_DRAFTER_PROMPT = "data/prompts/task_drafter.md"
# The generic version, used when the private prompt above is missing (public repository).
TASK_DRAFTER_PROMPT_EXAMPLE = "examples/prompts/task_drafter.md"

# --- Protecting the budget ---

# The most one run may spend, in US dollars. The run stops when spending reaches it.
# The check happens between calls, so a run can go over by the cost of one call.
MAX_RUN_COST_USD = 2.00

# A saved answer is reused instead of paying for the same call again, but only if it is
# at most this many days old. Providers update their models, so old answers go stale.
REUSE_ANSWERS_FOR_DAYS = 30

# Tokens assumed per word when estimating a run's cost before it starts. Set on the
# high side (the highest rate measured on the test tasks was about 2.2).
TOKENS_PER_WORD_ESTIMATE = 2.3

# --- Estimating the cost of a real job ---

# How many words one page is assumed to hold.
WORDS_PER_PAGE = 500

# The low and high cost figures are the estimate minus and plus this share (0.30 = 30%).
COST_UNCERTAINTY = 0.30

# --- Picking the best answer ---

# Fixed number that decides the shuffled order of answers in the web app (A, B, C, D), so
# the order is the same every time the page opens.
PICK_SHUFFLE_SEED = 0

# --- The judge ---

# The helper model that judges the answers. It is outside the comparison, so it cannot
# favour its own family.
JUDGE_MODEL = "qwen-3-8-27b"

# How many of your most recent picks are shown to the judge as examples of your standard.
JUDGE_EXAMPLES = 3

# Room for the judge's critiques and verdict, in tokens (kept low for Groq's free tier).
JUDGE_MAX_TOKENS = 900

# The judge's instructions. The private file is used if it exists, otherwise the generic
# one from examples/ (the same rule applies to the drafting prompt).
JUDGE_PROMPT = "data/prompts/judge.md"
JUDGE_PROMPT_EXAMPLE = "examples/prompts/judge.md"

# Your rules for the judge, one Markdown file per kind of task. A file named slide_*.md
# applies when the task mentions "slide"; a file named general_*.md always applies.
JUDGE_RULES_FOLDER = "data/judge"
