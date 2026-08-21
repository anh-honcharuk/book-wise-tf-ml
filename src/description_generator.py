from transformers import GPT2Tokenizer, TFGPT2LMHeadModel


class DescriptionGenerator:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            "gpt2-medium"
        )

        self.model = TFGPT2LMHeadModel.from_pretrained(
            "gpt2-medium",
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def generate_description(
        self,
        prompt,
        max_length=150,
    ):
        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors="tf",
        )

        sample_output = self.model.generate(
            input_ids,
            do_sample=True,
            max_length=max_length,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
        )

        return self.tokenizer.decode(
            sample_output[0],
            skip_special_tokens=True,
        )


if __name__ == "__main__":
    generator = DescriptionGenerator()

    prompt = (
        "A mysterious adventure story about a young hero "
        "who discovers a hidden world"
    )

    description = generator.generate_description(prompt)

    print("\nGenerated description:")
    print(description)