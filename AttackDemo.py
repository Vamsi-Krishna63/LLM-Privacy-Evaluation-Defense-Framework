#from data import JailbreakQueries
#from models import TogetherAIModels
#from attacks import Jailbreak
#from metrics import JailbreakRate

#data = JailbreakQueries()
# Fill api_key
#llm = TogetherAIModels(model="togethercomputer/llama-2-7b-chat", api_key="")
#attack = Jailbreak()
#results = attack.execute_attack(data, llm)
#rate = JailbreakRate(results).compute_metric()
#print("rate:", rate)


#------------------------------------------

# DeepSeek (via OpenRouter – 100% free, no card)

import openai
from data import JailbreakQueries
from attacks import Jailbreak
from metrics import JailbreakRate

# Step 1: Set OpenRouter API Key and Base URL
openai.api_key = "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"
openai.base_url = "https://openrouter.ai/api/v1"

# Step 2: Define OpenRouter-compatible Model Wrapper
class OpenRouterModel:
    def __init__(self, model_name="deepseek-chat"):
        self.model_name = model_name

    def query(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=256
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"Error: {e}"

# Step 3: Load Data, Run Jailbreak Attack, and Evaluate
if __name__ == "__main__":
    data = JailbreakQueries()
    llm = OpenRouterModel(model_name="deepseek-chat")  # You can change to another model if desired

    attack = Jailbreak()
    results = attack.execute_attack(data, llm)

    rate = JailbreakRate(results).compute_metric()
    print("Jailbreak Rate:", rate)
