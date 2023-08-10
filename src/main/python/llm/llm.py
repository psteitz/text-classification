"""
Zero-shot classification example using llama-2-70b-chat accessed via the Replicate API

Requires pip install replicate
Export Replicate API key as REPLICATE_API_TOKEN.
To get a key:  https://replicate.com.
Free usage is limited.
"""
import replicate
yahoo_classes = [
           "society or culture",
           "science or mathematics",
           "health",
           "education or reference",
           "computers or internet",
           "sports",
           "business or finance",
           "entertainment or music",
           "family or relationships",
           "politics or government"
]

categories_string = ""
for item in yahoo_classes:
    categories_string += item + "\n"

system_prompt = "Your job is to find the right category for each question and answer pair provided.\n \
The categories are: \n" + categories_string
                
def classify(question_anawer):
    output = replicate.run(
        "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
        input={"system_prompt": system_prompt, "prompt": question_anawer}
    )
    return output

def show_response(response):
    out = "response: "
    for item in response:
        out += item
    print(out)

print("System prompt: " + system_prompt)
print("Prompt: What elements make up water? Oygen and hydrogen.")
show_response(classify("What elements make up water? Oygen and hydrogen."))