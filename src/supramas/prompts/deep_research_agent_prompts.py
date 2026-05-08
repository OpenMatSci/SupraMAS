use_entities = True


def get_deep_regenerate_idea_prompt(input_data) -> str:
    final_idea = input_data["idea"]
    user_prompt = input_data["human_feedback"]
    prompt = f"""
You are a creative optimization expert. In our previous multi-turn conversation, we have generated an idea, but the user has provided some improvement suggestions (called `user_prompt`). Your task is to modify and optimize the original idea based on these suggestions to make it more aligned with the user's needs and expectations.

Original Idea:
{final_idea}

User Suggestions:
{user_prompt}

Please modify the idea according to the user's suggestions and explain the changes you made. If any part of the suggestion is unclear, feel free to make reasonable adjustments based on your understanding and experience.

The final idea should contains the title, clearly explain the origins, motivation, novelty, difference of previous works and challenges of your idea, detailing how you overcame these hurdles. 
Please output strictly in the following format:
<final_idea> {{the final idea}} </final_idea>
    """
    return prompt


def get_deep_refine_experiment_prompt(input_data) -> str:
    experiment = input_data["experiment"]
    human_feedback = input_data["human_feedback"]

    prompt=f"""
Your task is to modify a given experimental scheme for superconducting materials according to the suggestions provided by experts and give the reasons for the modifications.
First, please carefully read the following experimental scheme for superconducting materials:
<experiment_scheme>
{experiment}
</experiment_scheme>
Then, read the following suggestions from experts:
<expert_suggestions>
{human_feedback}
</expert_suggestions>
When modifying the experimental scheme, please follow these steps:
1. Thoroughly understand each suggestion from the experts.
2. Identify the parts in the experimental scheme that need to be modified based on the suggestions.
3. Make appropriate modifications to the experimental scheme according to the suggestions.
4. For each modification, think carefully about the reason behind it.
You MUST output **only** valid JSON inside the <experiment> ... </experiment> tag.
<experiment>
{{
    "modified_scheme": "The modified experimental scheme here",
    "reasons": "The reasons for the modifications here"
}}
</experiment>

"""
    
    return prompt
