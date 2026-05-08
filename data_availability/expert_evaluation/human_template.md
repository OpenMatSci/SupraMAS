Evaluate the generated research idea and detailed experimental plan for the given research topic. Assign scores and provide evidence grounded only in the supplied text.

Research topic:
<topic>
{topic}
</topic>

Research idea:
<idea>
{idea}
</idea>

Detailed experimental plan:
<experiment_plan>
{exp}
</experiment_plan>

---

Evaluation dimensions:
1. Content Completeness
   Assess whether the idea and the experimental plan contain the necessary scientific and procedural information.

2. Theme Relevance
   Assess whether the idea is relevant to the topic, whether the experimental plan is relevant to the idea, and whether the experimental plan is relevant to the topic.

3. Technical Innovation
   Assess whether the idea and experimental plan contain technically innovative elements, such as the material system, control strategy, synthesis route, characterization design or optimization logic.

4. Technical Feasibility
   Assess whether the proposed material system, synthesis route, parameters and characterization steps are experimentally feasible.

5. Application Value
   Assess whether the proposed idea and experimental plan have potential scientific or application value.

---

Output requirements:
1. Return one score and one evidence list for each of the five dimensions. Scores must be integers from 0 to 5.
2. Evidence must be based only on the supplied topic, idea and experimental plan. It may summarize the supplied text, but must not introduce unsupported information.
3. Return six fields in total: the five dimension-level evaluations plus one overall comment field. Leave comment as an empty string.
4. Wrap the JSON object in <template></template> tags and do not output unrelated text.

Required JSON template:
<template>
{{
    "Content Completeness": {{
        "score": 0,
        "evidence": [
            "Idea completeness: ...",
            "Experimental-plan completeness: ..."
        ]
    }},
    "Theme Relevance": {{
        "score": 0,
        "evidence": [
            "Topic-to-idea relevance: ...",
            "Idea-to-experimental-plan relevance: ...",
            "Topic-to-experimental-plan relevance: ..."
        ]
    }},
    "Technical Innovation": {{
        "score": 0,
        "evidence": [
            "Material-system innovation: ...",
            "Control-strategy innovation: ...",
            "Experimental-technique innovation: ..."
        ]
    }},
    "Technical Feasibility": {{
        "score": 0,
        "evidence": [
            "Material-system feasibility: ...",
            "Experimental-technique feasibility: ..."
        ]
    }},
    "Application Value": {{
        "score": 0,
        "evidence": [
            "..."
        ]
    }},
    "comment": ""
}}
</template>
