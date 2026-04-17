from groq import Groq


class Generator:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    def generate(self, query, context):
        prompt = f"""
        You are a supply chain analyst.

        STRICT RULES:
        - Use ONLY the provided context.
        - Do NOT assume missing data.
        - Do NOT invent formulas.
        - For averages, use simple mean:
          (value1 + value2 + ... ) / N
        - Do NOT use quantity-weighted averages.
        - If data is insufficient, say "Insufficient data".

        ---------------------
        EXAMPLES:

        Example 1:
        Context:
        router from wh-a (west) | qty=10, delay=5
        router from wh-b (west) | qty=20, delay=7

        Question:
        What is the average delivery time for routers in the west region?

        Answer:
        The average delivery time is (5 + 7) / 2 = 6 days.

        ---

        Example 2:
        Context:
        laptop from wh-x (east) | qty=5, delay=3
        laptop from wh-y (east) | qty=8, delay=5

        Question:
        Calculate the mean delay for laptops in the east region.

        Answer:
        The mean delay is (3 + 5) / 2 = 4 days.

        ---

        Incorrect approach (DO NOT DO THIS):
        (10 × 5 + 20 × 7) / (10 + 20) 

        Correct approach:
        (5 + 7) / 2 

        ---------------------

        NOW ANSWER:

        Context:
        {chr(10).join(context)}

        Question:
        {query}

        Answer:
        """

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",  
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content