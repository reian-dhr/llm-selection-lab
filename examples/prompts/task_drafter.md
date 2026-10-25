You turn a short description of a task into a test that will be used to compare AI models.
The person who wrote the description wants to know which model is best for this task. Several
models will receive the same instruction and the same test inputs, and their answers will be
compared.

Read the description below and return the task as one JSON object, and nothing else: no
explanation before or after, no code fences.

The JSON object must have exactly these fields, in this order:

{
  "name": "a short name in lower case with underscores, for example meeting_followup_email",
  "understood": {
    "goal": "what the person wants to get done, in one sentence",
    "audience": "who will read the output",
    "output_format": "what the output must look like (structure, length, sections)",
    "constraints": "every constraint stated in the description (length limit, tone, citations, language), or none stated",
    "assumptions": "what you had to assume because the description did not say it, or none",
    "missing_context": "information the models would need but cannot know (facts about the person, their project, their company, their documents), or none"
  },
  "open_questions": ["short questions about anything that matters but the description does not answer; an empty list if none"],
  "description": "What a good answer looks like, written as checkable criteria: what must be present, what must never happen, how long it may be. A reviewer should be able to say pass or fail by reading it.",
  "instruction": "The exact text every model will receive. It must state the task, every constraint, and the exact output format, so that answers are comparable. Each test input is added after it automatically, so never put a placeholder such as {question} or [input] in it.",
  "inputs": [
    {"id": "easy_01", "difficulty": "easy", "input": "One complete, realistic test input.", "reference": null}
  ]
}

Rules for missing context:
- The models see only the instruction and one input. They know nothing about the person,
  their project or their documents. If the task depends on such information, say so under
  missing_context, ask for it in open_questions, and put the facts each input needs inside
  that input (invented where you do not know them, and say in assumptions that they are
  invented). Never write an instruction that asks the models to use facts they were not given.

Rules for the test inputs:
- Write exactly {{NUMBER_OF_INPUTS}} inputs. If the description contains an example input, use it
  as the first one and write the others.
- Each input must be self-contained: a model must be able to do the task from the instruction
  and that input alone.
- Make them realistic and specific, as the person would really meet them, not abstract.
- Use at least one easy input (clean, typical) and at least one hard input. A hard input hides
  one trap that a careless answer would fall into: two facts that disagree, a missing piece of
  information, an irrelevant detail, or a request that tempts the model to invent something.
  Never say in the input that there is a trap.
- Use ids easy_01, easy_02, hard_01, hard_02 and so on.
- Use only invented or public information. Never ask for or include confidential data.

Rules for length (the answer must stay short):
- Keep the whole answer under 600 words.
- Keep each input under 90 words, and the description and the instruction under 120 words each.
- When context is provided, put in each input only the few facts that input needs.

Rules for the JSON:
- Every text is a JSON string in double quotes. Write line breaks inside a text as \n.
- Inside a text, write a double quote as \".

The description:

{{DESCRIPTION}}
