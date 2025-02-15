import re
import csv
import datetime
import time
import pyttsx3
import speech_recognition as sr
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import random
import os
import openai

class FeedbackAgent:
    def __init__(self, customer_data_file, conversation_log_file, openai_api_key):
        self.customer_data_file = customer_data_file
        self.conversation_log_file = conversation_log_file
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)
        self.engine.setProperty('rate', 160)
        self.customers = self.load_customer_data()
        self.recognizer = sr.Recognizer()
        self.sentiment_pipeline = pipeline("sentiment-analysis")
        self.topic_model_name = "facebook/bart-large-mnli"
        self.topic_tokenizer = AutoTokenizer.from_pretrained(self.topic_model_name)
        self.topic_model = AutoModelForSequenceClassification.from_pretrained(self.topic_model_name)
        self.topic_classifier = pipeline("zero-shot-classification", model=self.topic_model, tokenizer=self.topic_tokenizer)
        self.conversation_history = []  # Track the conversation flow
        self.conversation_active = True  # Flag to keep track of whether the converation is active or not
        openai.api_key = openai_api_key
        self.max_turns = 5  # Max number of conversation turns
        self.turns_taken = 0  # Number of turns taken so far
        self.topics_discussed = set()  # Track topics already discussed
        self.last_question_asked = None  # Track what question was asked last time
        self.openai_model = "gpt-3.5-turbo"  # The model you want to use

    def load_customer_data(self):
        customers = {}
        try:
            with open(self.customer_data_file, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    customers[row['customer_id']] = row
        except FileNotFoundError:
            print(f"Error: Customer data file '{self.customer_data_file}' not found.")
            return {}
        except Exception as e:
            print(f"Error loading customer data: {e}")
            return {}
        return customers

    def log_conversation(self, customer_id, agent_utterance, customer_utterance):
        timestamp = datetime.datetime.now().isoformat()
        with open(self.conversation_log_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([customer_id, timestamp, agent_utterance, customer_utterance])

    def speak(self, text):
        original_rate = self.engine.getProperty('rate')
        rate_variation = random.randint(-10, 10)
        volume_variation = random.randint(-5, 5)  # Volume
        self.engine.setProperty('rate', original_rate + rate_variation)
        self.engine.setProperty('volume', max(0, min(1, self.engine.getProperty('volume') + volume_variation / 100)))
        self.engine.say(text)
        self.engine.runAndWait()
        self.engine.setProperty('rate', original_rate)

    def record_audio(self, duration=10, phrase_time_limit=None):
        """Records audio from the microphone and saves it to a file."""
        filename = "customer_response.wav"  # Temporary audio file
        with sr.Microphone() as source:
            print("Recording...")
            self.recognizer.adjust_for_ambient_noise(source, duration=3)  # Ambient Adjustment
            audio = self.recognizer.listen(source, timeout=duration, phrase_time_limit=phrase_time_limit)  # Added the Time out and duration and phrase_time_limit
            with open(filename, "wb") as f:
                f.write(audio.get_wav_data())
        print("Finished recording.")
        return filename

    def transcribe_audio(self, audio_file, language="en"):
        """Transcribes an audio file to text using OpenAI Whisper API."""
        try:
            print("Transcribing...")  # Logging
            with open(audio_file, "rb") as audio_file_obj:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file_obj,
                    response_format="text",
                    language=language  # Add the language code here
                )
            text = transcript
            print(f"Transcription (Whisper): {text}")
            return text
        except Exception as e:
            print(f"Error transcribing with Whisper: {e}")
            return None
        finally:
            os.remove(audio_file)  # Delete the temp file

    def analyze_sentiment(self, text):
        print("Analyzing sentiment...")  # Logging
        result = self.sentiment_pipeline(text)[0]
        sentiment = result['label']
        score = result['score']
        return sentiment, score

    def extract_topics(self, text, candidate_labels):
        print("Extracting topics...")  # Logging
        result = self.topic_classifier(text, candidate_labels=candidate_labels)
        return result

    def detect_end_of_conversation(self, text, sentiment, score):
        """Detects if the customer wants to end the conversation."""
        text_lower = text.lower()

        # Regular expressions for more flexible matching
        end_patterns = [
            r"\b(thank you|that's all|i'm done|goodbye|no more|stop the call)\b",  # Word boundary to avoid partial matches
            r"end (the|this) call",
            r"i want to (end|stop) (this|the) conversation",
            r"please (end|stop)",
        ]

        for pattern in end_patterns:
            if re.search(pattern, text_lower):
                return True

        # Check for negative sentiment + negative keywords
        if sentiment == "NEGATIVE" and score > 0.7:  # Adjust the sentiment threshold as needed
            negative_keywords = ["unhappy", "disappointed", "frustrated", "wasted my time"]
            for keyword in negative_keywords:
                if keyword in text_lower:
                    return True

        return False

    def generate_follow_up_question(self, relevant_topics):
        """Generates a follow-up question using OpenAI based on conversation history and topics."""

        # Building prompt with the conversation history
        prompt = "You are a customer service agent calling for feedback about a customer's recent dining experience.  Based on these topics:" + ", ".join(relevant_topics) + ", generate ONLY a relevant follow-up question as if you were speaking directly to the customer.  Do not include any introductory phrases or explanations.  Just the question itself."

        # Appending converation history to the prompt to give context to the model
        prompt += "\n\nConversation History:\n"
        for turn in self.conversation_history:
            prompt += f"{turn['role']}: {turn['content']}\n"

        try:
            print("Generating follow-up question using OpenAI...")  # Logging

            response = openai.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful and friendly customer service agent.  Respond concisely and directly."},  # You can modify your ai bot
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50  # Adjust as needed
            )

            follow_up_question = response.choices[0].message.content.strip()
            # Remove any "Agent:" prefixes if they still exist.
            follow_up_question = follow_up_question.replace("Agent:", "").strip()  # Clean up the response
            print(f"Generated question: {follow_up_question}")
            return follow_up_question
        except Exception as e:
            print(f"Error generating follow-up question with OpenAI: {e}")
            return None

    def respond_to_sentiment(self, sentiment, score, customer_response):  # added the customer_response parameter
        """Responds to sentiment using OpenAI."""
        try:
            print("Generating response using OpenAI...")  # Logging

            prompt = f"You are a customer service agent. The customer provided the following feedback: {customer_response}.  The sentiment was {sentiment} with a score of {score}. Craft a positive and empathetic response as if speaking directly to the customer. Do not include 'Agent:' or any introductory phrases. Respond directly."

            # Appending converation history to the prompt to give context to the model
            prompt += "\n\nConversation History:\n"
            for turn in self.conversation_history:
                prompt += f"{turn['role']}: {turn['content']}\n"

            response = openai.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful, friendly, and empathetic customer service agent. Always respond in a positive and helpful tone. Be concise."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100  # Adjust as needed.
            )
            openai_response = response.choices[0].message.content.strip()
            openai_response = openai_response.replace("Agent:", "").strip()  # Clean up the response
            print(f"Generated response: {openai_response}")
            return openai_response

        except Exception as e:
            print(f"Error generating response with OpenAI: {e}")
            return "Thank you for your feedback. We appreciate it."

    def run_feedback_session(self, customer_id):
        customer = self.customers.get(customer_id)
        if not customer:
            print(f"Customer with ID {customer_id} not found.")
            self.speak(f"I'm sorry, I can't find a customer with ID {customer_id}.")
            return

        print(f"\nStarting feedback session for {customer['name']}...")
        greetings = [
            f"Hi {customer['name']}, thanks for taking my call! I'm calling about your recent visit on {customer['last_dine_date']}.",
            f"Hello {customer['name']}, how are you doing today? I'm following up on your dining experience on {customer['last_dine_date']}.",
            f"Good day, {customer['name']}! I hope you're having a good day. We're calling to get your feedback about your meal on {customer['last_dine_date']}."
        ]
        agent_utterance = random.choice(greetings) + " Could you tell me about it?"
        self.speak(agent_utterance)
        self.log_conversation(customer_id, "Agent: " + agent_utterance, "")
        self.conversation_history.append({"role": "agent", "content": agent_utterance})  # Store the conversation

        # Main conversation loop
        while self.conversation_active and self.turns_taken < self.max_turns:  # Condition to see if the turns are still active
            audio_file = self.record_audio()  # Record the audio file.
            customer_response = self.transcribe_audio(audio_file)  # Transcribe it.

            if customer_response:
                self.log_conversation(customer_id, "Customer Feedback: ", f"Customer said: {customer_response}")
                self.conversation_history.append({"role": "customer", "content": customer_response})

                sentiment, score = self.analyze_sentiment(customer_response)

                # Check if the customer wants to end the conversation
                if self.detect_end_of_conversation(customer_response, sentiment, score):  # added the sentiment and score
                    self.conversation_active = False
                    break  # Exit the loop

                candidate_labels = ["food quality", "service", "atmosphere", "pricing", "cleanliness", "staff friendliness", "waiting time"]
                topic_results = self.extract_topics(customer_response, candidate_labels)
                relevant_topics = [topic_results['labels'][i] for i in range(len(topic_results['labels'])) if topic_results['scores'][i] > 0.3]

                # Generate follow-up questions
                follow_up_question = self.generate_follow_up_question(relevant_topics)

                # Respond based on sentiment or ask follow-up question
                if follow_up_question:
                    self.speak(follow_up_question)
                    self.log_conversation(customer_id, "Agent: " + follow_up_question, "")
                    self.conversation_history.append({"role": "agent", "content": follow_up_question})
                    self.topics_discussed.update(relevant_topics)  # Topic was asked so we will now store it and never aske again
                    self.last_question_asked = follow_up_question
                else:
                    # No relevant topics, just respond to sentiment
                    response = self.respond_to_sentiment(sentiment, score, customer_response)  # added the customer_repsonse
                    self.speak(response)
                    self.log_conversation(customer_id, "Agent Response: ", response)
                    self.conversation_history.append({"role": "agent", "content": response})
                    self.speak("Is there anything else you would like to add?")  # open end question for the user
                    audio_end_test = self.record_audio(duration=5, phrase_time_limit=3)  # record the audio again to see if the user says yes or no.  # Shortening record_audio AND setting phrase_time_limit
                    customer_repsonse_end = self.transcribe_audio(audio_end_test)  # transcribe the audio again

                    if customer_repsonse_end:
                        customer_repsonse_end_lower = customer_repsonse_end.lower()
                        if "yes" in customer_repsonse_end_lower or "no" in customer_repsonse_end_lower:  # Checking for yes and no
                            self.conversation_active = False
                            break
                        else:  # They didn't say yes or no so it didn't understand
                            print("They have not respond with yes or no this will end the call since we have not understood.")
                            response = self.respond_to_sentiment(sentiment, score, customer_response)  # added the customer_repsonse
                            self.speak(response)
                            self.log_conversation(customer_id, "Agent Response: ", response)
                            self.conversation_history.append({"role": "agent", "content": response})
                            self.conversation_active = False
                            break
                self.turns_taken += 1  # Increment turns

            else:
                self.speak("Sorry, I had trouble understanding. Could you please repeat?")

        # End of conversation sequence
        ending_phrases = [
            "Thank you so much for your time and feedback. We really appreciate it!",
            "Thanks again for your feedback. We value your opinion.",
            "Have a great day!"
        ]
        self.speak(random.choice(ending_phrases))
        print("Feedback session complete.")

# Main execution
if __name__ == "__main__":
    customer_data_file = "customer_data.csv"
    conversation_log_file = "conversation_log.csv"
    openai_api_key = "dummy_api"  # Replace with your actual OpenAI API key

    agent = FeedbackAgent(customer_data_file, conversation_log_file, openai_api_key)
    agent.speak("Please enter the customer ID to connect with.")
    customer_id = input("Enter customer ID: ")
    agent.run_feedback_session(customer_id)
