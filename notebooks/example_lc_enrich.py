from langchain import HuggingFacePipeline


class LLMWrapper():

    def __init__(self, model="google/flan-t5-large", max_new_tokens=20):
        self.llm = HuggingFacePipeline.from_model_id(
            model_id=model,
            task="summarization",
            model_kwargs={"temperature": 0, "max_length": max_new_tokens},
        )

        # Total hack to find a parameter I could change easily with Transformers
        # self.llm.pipeline.model.config.max_new_tokens=max_new_tokens

        self.template = "{question}\n\nText: {context}\n\n Answer: "

    def query(self, question: str, context: str) -> str:
        """ Secret filter, max 300 characters of context. """
        query = self.template.format(question=question, context=context[:500])
        return self.llm.predict(query)


if __name__ == "__main__":
    llm = LLMWrapper()

    print(llm.query(
        question="What is the name of the antagonist in this text?",
        context="Mary threw a snowball at Andrew"
    ))
