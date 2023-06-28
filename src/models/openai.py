import openai

openai.organization = "org-DlbzLzT5CDhoGm9GuaIkleGI"
openai.api_key = os.getenv("OPENAI_API_KEY")

class ChatGptInterface:
    def response(cls, input_text, top_k:int=None, do_sample:bool=False, max_new_tokens:int=None):
        temp = 1 if do_sample else 0
        output = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-0301',
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text},
            ],
            temperature=temp, # 0.0 = deterministic
            max_tokens=max_new_tokens, # max_tokens is the generated one,
        )
        return output
    