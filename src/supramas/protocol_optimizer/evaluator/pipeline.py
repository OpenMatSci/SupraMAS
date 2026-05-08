import re

from supramas.utils import extract
from .agents import Challenger, Checker


class EvaluatorPipeline:
    def __init__(self, retrival, exp_plan, llm):
        self.challenger = Challenger(retrival=retrival,llm = llm)
        self.checker = Checker(retrival=retrival,llm = llm)
        self.retrival = retrival
        self.exp_plan = exp_plan
        self.llm = llm

    def judge(self, challenge_idea, check_idea):
        suggestion_challenge = extract(challenge_idea, "suggestion")
        suggestion_check = extract(check_idea, "suggestion")
        judge_challenge = True if extract(challenge_idea, "judge") == "True" else False
        judge_check = True if extract(check_idea, "judge") == "True" else False
        ### situation 1
        if judge_challenge and judge_check:
            return None
        ### situation 2
        else:
            suggestion = f"The opinions of the skeptics experts are as follows:{suggestion_challenge} \n The opinions of the audit experts are as follows:{suggestion_check}"
            return suggestion

    def extract_sub_experiments(self, text):
        pattern = re.compile(r"<sub_experiment_(\d+)>\s*(.*?)\s*</sub_experiment_\1>", re.DOTALL)
        experiments = []
        for match in pattern.findall(text):
            id, content = match
            experiments.append({"id": id, "content": content.strip()})
        print("experiments", experiments)
        return experiments

    # 从子实验内容中提取特定标签（如 judge, suggestion）
    def extract_tag_from_sub(self, content, tag):
        pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL)
        match = pattern.search(content)
        return match.group(1).strip() if match else None

    # 对 challenge_idea 和 check_idea 中每个子实验逐个对比并汇总建议
    def judge_experiment(self, challenge_idea, check_idea):
        # challenge_subs = self.extract_sub_experiments(challenge_idea)
        # check_subs = self.extract_sub_experiments(check_idea)

        # 保证编号对齐，一一对应
        assert len(challenge_idea) == len(check_idea), "Mismatch in sub-experiment count."

        all_passed = True
        suggestions = []

        for idx,(ch, ck) in enumerate(zip(challenge_idea.values(), check_idea.values())):
            ch_judge = ch["judge"] == True
            ck_judge = ck["judge"] == True

            if not (ch_judge and ck_judge):
                all_passed = False
                ch_suggestion = ch["suggestion"]
                ck_suggestion = ck["suggestion"]
                suggestions.append(
                    {
                        "idx": idx,
                        "feedback": f"The opinions of the skeptics experts are as follows:{ch_suggestion.strip()}\n"
                        f"The opinions of the audit experts are as follows:{ck_suggestion.strip()}",
                    }
                )

        return None if all_passed else suggestions

    def forward(self, plan, type="idea"): # 评估的plan是idea或者是experiment
        if type=="idea":
            Challenge_idea = self.challenger.get_challenge_idea(plan)
            print("Challenge_idea\n", Challenge_idea)

            Check_idea = self.checker.get_check_idea(plan)
            print("Check_idea\n", Check_idea)
            suggestions = self.judge(Challenge_idea, Check_idea)

        elif type=="experiment":
            Challenge_idea = self.challenger.get_challenge_experiment(plan, self.exp_plan.steps)
            # print("Challenge_idea\n",Challenge_idea)

            Check_idea = self.checker.get_check_experiment(plan, self.exp_plan.steps)
            # print("Check_idea\n", Check_idea)

            suggestions = self.judge_experiment(Challenge_idea, Check_idea)

        return suggestions


# if __name__ == "__main__":
#     from vector_retrival import VectorRetrieval
#     from supramas.utils import *
#     retrival = VectorRetrieval()
#     evaluator = EvaluatorPipeline(retrival=retrival)
#     idea = "The new technology can improve the efficiency of the company's production line."
#     is_pass,suggestion = evaluator.forward(idea)
#     print(is_pass,suggestion)
