from google import genai

client = genai.Client(api_key="AIzaSyByqmzfmHQlP2_CRADbwZmFGFBcxu5zSDQ")

response = client.models.generate_content(
    model="gemini-3-pro-preview",
    contents="Hello, how are you?",
)

print(response.text)
