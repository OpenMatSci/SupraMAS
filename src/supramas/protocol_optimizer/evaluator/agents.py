from supramas.utils import extract, extract_sub_experiments
from openai import OpenAI
import json

# from supramas.utils import *


class Challenger:
    def __init__(self, retrival,llm):
        """
        a class for assessing the feasibility and logical coherence of the overall approach,
        and propose improvement suggestions for the identified challenges.
        """
        self.retrival = retrival
        self.llm = llm

    def get_challenge_experiment(self, plan, sub_experiments) -> str:
        """
        Args:
            plan (str): final experiment

        Returns:
            str: suggestions for improving the sub_experiments
        """
        description = self.get_description(plan)

        stucture = self.get_structure(plan)

        retrieved_contents = self.retrival([description])

        challenge_experiment = self.challenge_experiment(retrieved_contents, description, stucture, sub_experiments)

        return challenge_experiment

    def challenge_experiment(self, retrieved_contents, description, structure, sub_experiments) -> str:
        # 组成prompt
        prompt = f"""
You are an expert research evaluator. I will provide you with a research plan that contains multiple interrelated sub-experiments or sub-modules.  
Each sub-experiment is designed to test or validate a specific aspect of the overall hypothesis or design. **Crucially, you must evaluate each sub-experiment not in isolation, but with full consideration of its relationships, dependencies, and interactions with the other sub-experiments.**

Here is the information for your evaluation:

- **Overall experiment description**:  
{description}

- **Sub-experiments**:  
{sub_experiments}

- **Structured framework** (including research objectives, innovation, and challenging aspects):  
{structure}

- **Reference templates** (examples of sound experimental designs or theoretical applications):  
{retrieved_contents}

For each sub-experiment, assess it based on the following three criteria:
1. **Research Objective**: Is it clear, meaningful, and feasible?
2. **Innovation**: Does it demonstrate novelty in theory, methodology, or application?
3. **Challenges & Issues**: What are the potential problems, and how could it be improved?

After comprehensive evaluation:
- If a sub-experiment has no serious flaws and requires no modification, set `judge` to `true`.
- If any improvement is needed (including issues arising from poor integration with other sub-experiments), set `judge` to `false`, and provide actionable suggestions.

**Output Format Requirements**:
- Output **only one** top-level JSON object inside `<challenge>` tags.
- The JSON keys must be `sub_experiment_1`, `sub_experiment_2`, ..., matching the order and numbering of the input sub-experiments.
- Each value is an object with two keys:  
- `"judge"`: a boolean (`true` or `false`)  
- `"suggestion"`: a concise, constructive string (even if `judge` is `true`, include a brief note like "No major issues detected." or "Objectives are clear and well-integrated.")

**Strictly follow this output format**:

<challenge>
{{
"sub_experiment_1": {{
    "judge": true,
    "suggestion": "No major issues detected."
}},
"sub_experiment_2": {{
    "judge": false,
    "suggestion": "The objective overlaps with sub-experiment 1; consider merging or clarifying distinct contributions."
}}
}}
</challenge>

Do not include any additional text, explanation, or formatting outside the <challenge> tags.

            """

        # 调用大模型
        messages = self.wrap_messages(prompt)
        conversation_kwargs = {
            "stage": "plan_revision",
            "role": "challenge_agent",
            "meta": {
                "input_data": {"description": description, "structure": structure, "retrieved_contents": retrieved_contents, "sub_experiments": sub_experiments},
                "prompt": messages,
            }
        }
        response_content = self.llm.response(messages,extract_key="challenge",load_json=True,**conversation_kwargs)

        return response_content

    def get_challenge_idea(self, idea) -> str:
        """
        Args:
            idea (str): a research idea

        Returns:
            str: suggestions for improving the idea
        """
        description = self.get_description(idea)
        stucture = self.get_structure(idea)
        retrieved_contents = self.retrival([description])
        challenge_idea = self.challenge_idea(retrieved_contents, description, stucture)
        return challenge_idea

    def challenge_idea(self, retrieved_contents, description, structure) -> str:
        # 组成prompt
        prompt = f"""
        You are an expert reviewer of scientific research proposals. I will provide you with:

        - A research idea consisting of an experimental design or theoretical application, along with a brief introduction:  
        {description}

        - A structured framework outlining the research objectives, innovation points, and challenging aspects:  
        {structure}

        - Several reference templates of high-quality experimental designs or theoretical applications:  
        {retrieved_contents}

        Each reference template includes:
        - A `reason_flow` section: describing the logical reasoning, evaluation criteria, and critical considerations.
        - An `example_application` section: providing concrete implementation details.

        **Your task is to focus primarily on the `reason_flow` sections of the reference templates** to critically evaluate the provided research idea as a whole, addressing the following three aspects:

        1. **Research Objectives**:  
        Are they clearly stated, theoretically or practically valuable, and logically consistent with the proposed approach?  
        Is the objective feasible, and can the described methodology realistically achieve it?

        2. **Innovation**:  
        Does the idea introduce genuine novelty—such as new theoretical frameworks, methodological improvements, consideration of previously ignored physical effects, or novel applications?

        3. **Challenges & Improvement Opportunities**:  
        Based on the above critiques, identify unresolved issues or weaknesses in the objectives, innovation claims, or handling of challenges mentioned in the structured framework. Propose concrete improvements.

        After comprehensive evaluation:
        - If the idea is sound, well-justified, feasible, and requires no significant revision, set `judge` to `true`.
        - If any critical flaw or improvement opportunity exists (even if minor), set `judge` to `false`.

        **Output Requirements**:
        - Output **only** a single JSON object wrapped in `<challenge>` tags.
        - The JSON must contain exactly two keys:  
        - `"judge"`: a boolean (`true` or `false`)  
        - `"suggestion"`: a concise but comprehensive string addressing improvements for research objectives, innovation, and/or challenges (even if `judge` is `true`, include a brief affirmation like "The idea is well-structured and feasible.")

        **Strictly follow this format**:

        <challenge>
        {{
        "judge": false,
        "suggestion": "The research objective lacks specificity in measurable outcomes. The claimed innovation overlaps with prior work in [Reference X]; consider clarifying the novel theoretical contribution. Additionally, the challenge of [specific issue] is acknowledged but not addressed methodologically."
        }}
        </challenge>

        Do not include any additional text, explanations, or formatting outside the <challenge> tags.
        """

        # 调用大模型
        messages = self.wrap_messages(prompt)
        response_content = self.llm.response(messages,extract_key="challenge")
        # response_content = json.loads(response_content)

        return response_content

    def get_description(self, plan) -> str:
        prompt = f"""
you are provided with a research plan, summarize it into description in two scenarios. 
If the main focus of this plan is on describing experimental methods, summarize it as follows: 
state what was done in which material system, what effect was achieved, explain the reasons behind this effect, and clarify which theories or physical phenomena underlie it. 
The summary must strictly follow the experimental plan.
All points must be derived exclusively from the plan.
the research plan you are provided with is:{plan}, then get it into a description.

Please output strictly in the following format:
<description> {{description}} </description>
                 """
        messages = self.wrap_messages(prompt)
        conversation_kwargs = {
            "stage": "plan_revision",
            "role": "structure_agent",
            "meta": {
                "input_data": plan,
                "prompt": messages,
            }
        }
        response_content = self.llm.response(messages,extract_key="description",**conversation_kwargs)

        return response_content

    def get_structure(self, idea) -> str:
        prompt = f"""
you are provided with an research plan:{idea}, please summarize it into Research Objectives, Innovations and Challenging Aspects:
Research Objectives: What problem is the study aiming to solve in the future or what effect is it intended to achieve?
Innovations: Novel theories/physical models guiding the experiment or new experimental methods employed.
Challenging Aspects: Unresolved challenges or difficulties in the proposed experimental design.
Finally, present the research objectives, innovations, and challenges as distinct sections in a structured summary.
All points must be derived exclusively from the research plan.

Please output strictly in the following format:
<structure> {{structured summary}} </structure>
                 """
        messages = self.wrap_messages(prompt)
        conversation_kwargs = {
            "stage": "plan_revision",
            "role": "structure_agent",
            "meta": {
                "input_data": idea,
                "prompt": messages,
            }
        }
        response_content = self.llm.response(messages,extract_key="structure",**conversation_kwargs)

        return response_content

    def wrap_messages(self, prompt):
        messages = [
            {"role": "system", "content": "You are a physicist"},
            {"role": "user", "content": prompt},
        ]
        return messages


class Checker:
    def __init__(self, retrival,llm):
        """
        a class for examining the specific knowledge points involved in the research plan,
        output on which steps in the experimental design contain knowledge-based errors,
        and clarify what the correct or better approaches should be.
        """
        # self.milvus_client = MilvusClient(uri=os.getenv("MILVUS_URI"),user=os.getenv("MILVUS_USER"),password=os.getenv("MILVUS_PASSWORD"),db_name=os.getenv("MILVUS_DATABASE"))
        self.retrival = retrival
        self.llm = llm

    def get_check_experiment(self, plan, sub_experiments) -> str:
        """
        Args:
            plan (str): a research idea

        Returns:
            str: suggestions for improving the idea
        """
        description = self.get_description(plan)
        exps_detail = self.get_details_experiment(plan)
        knowledge_with_templates = []
        for key,detail in exps_detail.items():
            detail = detail["details"]
            retrieved_content = self.retrival([detail])
            print("retrieved_content", retrieved_content)
            knowledge_with_templates.append({f"knowledge_chain": detail, "retrieved_content": retrieved_content})

        check_experiment = self.check_experiment(description, knowledge_with_templates, sub_experiments)

        return check_experiment

    def check_experiment(self, description, knowledge_with_templates, sub_experiments) -> str:
        # 组成prompt
        prompt = f"""
        You are an expert scientific evaluator. You will be given a research plan composed of multiple interrelated sub-experiments or sub-modules.  
        These sub-experiments are **not independent**—each contributes to and is logically constrained by the others, forming a cohesive whole.

        Here is the information provided:

        - **Overall research plan description**:  
        {description}

        - **Sub-experiments** :  
        {sub_experiments}

        - **Reference knowledge and templates**:  
        {knowledge_with_templates}  

        Each reference entry includes:
        - `knowledge_chain`: the specific physical theories, phenomena, or process parameters that underpin a valid approach.
        - `retrieved_content`: a structured template of a sound experimental design or theoretical application, serving as a benchmark.

        **Your task**:
        Evaluate each sub-experiment **individually**, but with awareness of its role in the integrated system. For each one:
        1. Cross-check whether it correctly incorporates the relevant elements from the provided `knowledge_chain` (e.g., physical theories, parameters, mechanisms).
        2. Use the `retrieved_content` as a reference to assess methodological soundness and suggest improvements if needed.
        3. Provide numbered improvement suggestions (e.g., "1. ... 2. ...") only if modifications are required.

        After evaluation:
        - If a sub-experiment is well-aligned with the knowledge chain, methodologically sound, and consistent with the overall plan, set `judge` to `true`.
        - If any misalignment, omission, or design flaw exists (including poor integration with other sub-experiments), set `judge` to `false` and provide actionable suggestions.

        **Output Requirements**:
        - Output **only one JSON object**, wrapped strictly in `<check>` tags.
        - JSON keys must be `"sub_experiment_1"`, `"sub_experiment_2"`, ..., matching the numbering and order of the input.
        - Each value is an object with:
        - `"judge"`: a boolean (`true` or `false`)
        - `"suggestion"`: a string. If `judge` is `true`, write a brief confirmation (e.g., "No issues detected; aligns well with knowledge chain and overall design."). If `false`, provide numbered suggestions.

        **Strictly follow this format**:

        <check>
        {{
        "sub_experiment_1": {{
            "judge": true,
            "suggestion": "No issues detected; aligns well with knowledge chain and overall design."
        }},
        "sub_experiment_2": {{
            "judge": false,
            "suggestion": "1. The thermal expansion coefficient is not accounted for in the material selection, contradicting the knowledge_chain on temperature-dependent deformation. 2. Consider adopting the calibration protocol from retrieved_content to improve measurement accuracy."
        }}
        }}
        </check>

        Do not output any text outside the <check> tags. Do not use XML, markdown, or additional fields.
        """

        # 调用大模型
        messages = self.wrap_messages(prompt)
        conversation_kwargs = {
            "stage": "plan_revision",
            "role": "review_agent",
            "meta": {
                "input_data": {"description": description, "knowledge_with_templates": knowledge_with_templates, "sub_experiments": sub_experiments},
                "prompt": messages,
            }
        }
        response_content = self.llm.response(messages,extract_key="check",load_json=True,**conversation_kwargs)
        # response_content = json.loads(response_content)

        return response_content

    def get_check_idea(self, idea) -> str:
        """
        Args:
            idea (str): a research idea

        Returns:
            str: suggestions for improving the idea
        """
        description = self.get_description(idea)
        details = self.get_details(idea)
        retrieved_contents = self.retrival([description])
        check_idea = self.check_idea(retrieved_contents, description, details)
        return check_idea

    def check_idea(self, retrieved_contents, description, details) -> str:
        # 组成prompt
        prompt = f"""
        You are a scientific reviewer specializing in experimental and theoretical research validation. You are given:

        - A research idea (experimental design or theoretical application) with a brief introduction:  
        {description}

        - The specific knowledge points and knowledge chain that underpin this idea:  
        {details}

        - Several reference templates of high-quality designs or applications:  
        {retrieved_contents}

        Each reference template includes:
        - A `reason_flow`: explaining the logical reasoning, evaluation criteria, and scientific considerations.
        - An `example_application`: providing concrete implementation details.

        **Your task**:
        Using the `reason_flow` sections of the reference templates as analytical guides, critically evaluate the provided **knowledge chain and individual knowledge points** for:
        - Scientific correctness,
        - Logical consistency,
        - Methodological soundness,
        - Potential for improvement (e.g., better parameter control, more robust theoretical grounding, or alternative approaches suggested by the reference templates).

        If issues are found, provide **numbered improvement suggestions** (e.g., "1. ... 2. ...") that directly address the knowledge chain or specific knowledge points.

        After evaluation:
        - If the knowledge chain is accurate, complete, and well-supported by the idea and references, set `judge` to `true`.
        - If any correction, refinement, or enhancement is needed, set `judge` to `false`.

        **Output Requirements**:
        - Output **only one JSON object**, strictly wrapped in `<challenge>` tags.
        - The JSON must contain exactly two keys:
        - `"judge"`: a boolean (`true` or `false`)
        - `"suggestion"`: a string.  
            - If `judge` is `true`, write a brief confirmation (e.g., "The knowledge chain is scientifically sound and consistent with reference standards.")  
            - If `judge` is `false`, provide a numbered list of concrete, actionable suggestions (e.g., "1. The assumption of isotropic diffusion contradicts the anisotropic crystal structure mentioned in the description. 2. Consider incorporating temperature-dependent permittivity as shown in retrieved_content template 3.")

        **Strictly follow this format**:

        <challenge>
        {{
        "judge": false,
        "suggestion": "1. The knowledge point on electron-phonon coupling lacks quantitative parameterization. 2. The chain assumes steady-state conditions, but the experimental setup implies transient dynamics—revise to include time-resolved analysis as in retrieved_content example."
        }}
        </challenge>

        Do not include any additional text, explanations, XML tags, or formatting outside the <challenge> tags.
        """

        # 调用大模型
        messages = self.wrap_messages(prompt)
        response_content = self.llm.response(messages,extract_key="challenge")
        response_content = json.loads(response_content)

        return response_content

    def get_description(self, plan) -> str:
        prompt = f"""
            you are provided with a research plan, summarize it into description in two scenarios. 
            If the main focus of this plan is on describing experimental methods, summarize it as follows: 
            state what was done in which material system, what effect was achieved, explain the reasons behind this effect, and clarify which theories or physical phenomena underlie it. 
            The summary must strictly follow the experimental plan.
            All points must be derived exclusively from the plan.
            the research plan you are provided with is:{plan}, then get it into a description.

            Please output strictly in the following format:
            <description> {{description}} </description>
                 """
        messages = self.wrap_messages(prompt)
        conversation_kwargs = {
            "stage": "plan_revision",
            "role": "description_agent",
            "meta": {
                "input_data": plan,
                "prompt": messages,
            }
        }
        response_content = self.llm.response(messages,extract_key="description",**conversation_kwargs)
        
        return response_content

    def get_details_experiment(self, plan) -> str:
        prompt = f"""
        You are a scientific reasoning assistant. You will be given a research plan (an experimental design or theoretical application idea).  

        Your task is to:
        1. **Decompose** the overall experimental design into a sequence of procedural steps—up to a maximum of 10 logically ordered sub-steps.
        2. For **each sub-step**, construct a "knowledge chain" that:
        - Identifies the specific **physical theories**, **physical phenomena**, or **process parameters** involved,
        - Explains **how the experimental design of that sub-step explicitly accounts for, utilizes, or controls** the influence of those elements.
        3. **All knowledge must be derived exclusively from the provided idea**—do not infer or add external scientific knowledge.

        The input research plan is:
        {plan}

        **Output Requirements**:
        - Output **only one JSON object** wrapped in `<knowledge_chain>` tags.
        - The JSON keys must be `"sub_experiment_1"`, `"sub_experiment_2"`, ..., up to `"sub_experiment_N"`, in strict numerical order.
        - Each value is an object with a single key `"details"` whose value is a concise, self-contained paragraph describing the knowledge chain for that step.
        - If the idea cannot be meaningfully decomposed into steps, output an empty JSON object: `{{}}`.

        **Strictly follow this format**:

        <knowledge_chain>
        {{
        "sub_experiment_1": {{
            "details": "Step 1 involves calibrating the laser intensity based on the photoelectric effect theory; the design adjusts photon energy to match the work function of the target material, ensuring electron emission is controllable."
        }},
        "sub_experiment_2": {{
            "details": "Step 2 measures carrier mobility under varying electric fields, accounting for Drude model predictions by isolating scattering effects through low-temperature operation."
        }}
        }}
        </knowledge_chain>

        Do not include any additional text, markdown, or explanation outside the <knowledge_chain> tags.
        """
        messages = self.wrap_messages(prompt)
        conversation_kwargs = {
            "stage": "plan_revision",
            "role": "knowledge_chain_agent",
            "meta": {
                "input_data": plan,
                "prompt": messages,
            }
        }

        response_content = self.llm.response(messages,extract_key="knowledge_chain",load_json=True,**conversation_kwargs)
        # response_content = json.loads(response_content)
        
        return response_content

    def get_details(self, idea) -> str:
        prompt = f"""
you are provided with a research plan, decompose the overall experimental design of the idea into multiple procedural steps as much as possible. 
Then summarize the corresponding knowledge points for each sub-step of total experimental design in the form of a knowledge chain: 
describe the physical theories, physical phenomena, or process parameters involved in each experimental step, 
and specify how the experimental design of a particular sub-step accounts for the influence of a specific physical theory, physical phenomenon, or process parameter. 
All knowledge points must be derived exclusively from the idea.

the research plan you are provided with is:{idea},
then, arrange the decomposed details in numerical sequence: 1, 2, 3, 4...
               
Please output strictly in the following format:
<details> {{details}} </details>
                 """
        messages = self.wrap_messages(prompt)
        response_content = self.llm.response(messages,extract_key="details")
        response_content = json.loads(response_content)

        return response_content

    def wrap_messages(self, prompt):
        messages = [
            {"role": "system", "content": "You are a physicist"},
            {"role": "user", "content": prompt},
        ]
        return messages


class Judgement:
    def __init__(self, retrival):
        self.retrival = retrival

    def judge(self, challenge_idea, check_idea):
        suggestion_challenge = extract(challenge_idea, "suggestion")
        suggestion_check = extract(check_idea, "suggestion")
        judge_challenge = extract(challenge_idea, "judge")
        judge_check = extract(check_idea, "judge")
        ### situation 1
        if judge_challenge == True and judge_check == True:
            return True
        ### situation 2
        if judge_challenge == False or judge_check == False:
            suggestion = {
                "suggestionthe for the feasibility and logical coherence of the overall approach": suggestion_challenge,
                "suggestion for specific knowledge points": suggestion_check,
            }
            return suggestion
