from llama_cpp import Llama

# Path to your downloaded SmolLM2 model (GGML .bin file)
model_path = r"C:\Users\PARTH\Desktop\models\smollm2-135M.ggmlv3.q4_0.bin"

# Load the model
llm = Llama(model_path=model_path)

# Run inference
output = llm("Explain gravity to a class 7 student in simple terms.")
print(output)
