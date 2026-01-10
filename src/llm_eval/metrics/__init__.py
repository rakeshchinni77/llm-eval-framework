# Import ALL metrics so they self-register

# Reference metrics
from llm_eval.metrics.reference.bleu import BLEUMetric
from llm_eval.metrics.reference.rouge_l import RougeLMetric
from llm_eval.metrics.reference.bertscore import BERTScoreMetric

# RAG metrics
from llm_eval.metrics.rag.faithfulness import FaithfulnessMetric
from llm_eval.metrics.rag.context_relevancy import ContextRelevancyMetric
from llm_eval.metrics.rag.answer_relevancy import AnswerRelevancyMetric

# Judge metric
from llm_eval.metrics.judge.llm_judge import LLMJudgeMetric
