from google import genai

client = genai.Client(
    api_key="AIzaSyC6YMHjlZ4wHE4z2h2ystofHQ6o_HKzXeA"
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello"
)

print(response.text)