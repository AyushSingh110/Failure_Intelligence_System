from engine.pipeline.realtime_pipeline import run_realtime_pipeline

prompt = input("Enter your prompt: ")

result = run_realtime_pipeline(prompt)

print("\nFinal Verification Result:")
print(result)