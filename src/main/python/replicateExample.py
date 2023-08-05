"""
Zero-shot classification example using the Replicate API
"""
import replicate
categories = ["Science", "Sports", "Relationships"] 

categories_string = ""
for item in categories:
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

show_response(classify("How do chickens reporduce?  They lay eggs."))

