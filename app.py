import streamlit as st
import chromadb
import pickle
import google.generativeai as genai
from dotenv import load_dotenv
import os
import re
import json
from datetime import datetime
from chromadb.config import Settings
import time

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

client = chromadb.Client(Settings(anonymized_telemetry=False, persist_directory="./chroma_db"))
collection = client.get_or_create_collection(name="folio_knowledge_base")

with open("folio_chunks.pkl", "rb") as f:
    data = pickle.load(f)

if collection.count() == 0:
    for i, (doc, emb) in enumerate(zip(data["docs"], data["embeddings"])):
        collection.add(ids=[f"doc_{i}"], documents=[doc], embeddings=[emb])

st.set_page_config(page_title="Folio Chatbot", layout="centered")
st.title("📚 Folio Chatbot (Gemini RAG)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "contact_info" not in st.session_state:
    st.session_state.contact_info = []


def validate_contact_info(text):
    text = text.strip().lower()

    # Early discard of common non-contact context
    non_contact_keywords = [
        "pages", "page", "words", "copies", "chapters", "characters", "volumes",
        "years", "months", "age", "draft", "deadline", "book", "novel", "story",
        "genre", "formatting", "edits", "illustrations", "cover", "horror", "mystery"
    ]
    if any(word in text for word in non_contact_keywords):
        return "invalid"

    # Discard genre + number combos like "500-page horror novel"
    genre_keywords = ["horror", "romance", "thriller", "fantasy", "fiction", "nonfiction", "memoir"]
    if any(genre in text for genre in genre_keywords) and re.search(r"\b\d{1,4}\b", text):
        return "invalid"

    # Email detection
    email_regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    email_match = re.search(email_regex, text)
    if email_match:
        email = email_match.group(0)
        if any(bad in email.split('@')[0] for bad in ['example', 'test', 'sample', 'demo', 'domain', 'fake']):
            return "invalid_email"
        if text.endswith('@') or text.endswith('.'):
            return "invalid_email"

        contact_phrases = ["my email is", "email me at", "here's my email", "you can email", "reach me at", "contact me at", "email:"]
        if any(phrase in text for phrase in contact_phrases):
            return "valid_email"
        else:
            return "ambiguous_email"

    #phone detection
    phone_digits = re.sub(r"[^\d]", "", text)

    if 10 <= len(phone_digits) <= 15:
        phone_phrases = ["my phone is", "call me at", "text me at", "reach me on", "contact number", "phone:", "here’s my number"]
        
        #avoid misinterpreting common years
        if re.search(r"\b(19|20)\d{2}\b", text) and not any(phrase in text for phrase in phone_phrases):
            return "invalid"
        
        if any(phrase in text for phrase in phone_phrases):
            return "valid_phone"
        else:
            return "ambiguous_phone"

    elif phone_digits:
        #avoid short or story-related numbers
        if len(phone_digits) < 7:
            return "invalid"

        narrative_clues = ["since", "when", "was", "were", "at the age of", "in grade", "for", "years old", "back in"]
        if any(phrase in text for phrase in narrative_clues):
            return "invalid"

        return "invalid_phone"

    return "invalid"

real_person_phrases = [
    "real person", "connect me with a real person", "are you a real person",
    "can I speak to a human", "talk to someone", "talk to a human", "real human"
]

sample_phrases = [
    "cover samples", "cover examples", "sample covers", "book cover examples",
    "show me covers", "can I see designs", "any samples", "can I see examples",
    "sample book design", "examples of your design"
]

if st.button("🧹 Reset Chat"):
    st.session_state.chat_history = []
    st.session_state.contact_info = []
    st.rerun()

query = st.chat_input("Ask me anything about publishing...")

if query:
    query = query.strip()
    st.chat_message("user").write(query)
    st.session_state.chat_history.append({"role": "user", "content": query})

    # handle real person request
    if any(phrase in query.lower() for phrase in real_person_phrases):
        response_text = "You're chatting with a real person. How can I help you today?"
        st.chat_message("assistant").write(response_text)
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        st.stop()

    validation_result = validate_contact_info(query)
    user_name = st.session_state.get("user_name", "")

    if validation_result != "invalid":
        if validation_result == "invalid_email":
            response_text = "That doesn’t look like a valid email. Feel free to double-check it, no rush!"
        elif validation_result == "invalid_phone":
            response_text = "That doesn't seem like a full phone number, but no worries if you're not ready to share!"
        elif validation_result == "ambiguous_email":
            response_text = "That kind of looks like an email, were you trying to share your contact info?"
        elif validation_result == "ambiguous_phone":
            response_text = "Looks like a phone number, just checking, are you sharing it so we can reach out?"
        elif validation_result == "valid_email":
            #extract email from input
            email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", query)
            email = email_match.group(0) if email_match else query
            st.session_state.contact_info.append({"email": email})
            response_text = f"Thanks{', ' + user_name if user_name else ''}! I’ve saved your email. I’ll reach out shortly to set up a call and go over everything."
        elif validation_result == "valid_phone":
            #extract phone digits from input
            phone_digits = re.sub(r"[^\d+]", "", query)
            st.session_state.contact_info.append({"phone": phone_digits})
            response_text = f"Perfect{', ' + user_name if user_name else ''}! I’ve saved your number. We’ll contact you soon to schedule a call."
        else:
            #fallback, just store raw text
            st.session_state.contact_info.append(query)
            response_text = "Thanks for sharing your contact info!"

        st.chat_message("assistant").write(response_text)
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        st.stop()

    #embed and retrieve
    embed = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="retrieval_query"
    )
    results = collection.query(query_embeddings=[embed["embedding"]], n_results=1)
    context = results["documents"][0][0]
    chat_transcript = "\n".join([msg["content"] for msg in st.session_state.chat_history])

    full_prompt = f"""
You are Becca Williams — a friendly, professional support agent from Folio Publishers. Always use this name and never any other.

Your main services to offer based on user needs:
- Ghostwriting or writing help if they want someone to write their book.
- Book editing if they mention editing or proofreading.
- Book design if they ask about covers, layouts, or design samples.
- Marketing if they want help promoting their book.

Conversation guidelines:

1. Always listen carefully and respond naturally by:
   - Acknowledging their specific request or problem.
   - Suggesting the relevant Folio service(s).
   - Asking a relevant follow-up question to better understand their project.
   - Casually asking for their first name after the first or second reply to personalize the chat.

2. Do NOT assume the user is an author unless they explicitly say so.
   - If they say "someone to write my book," suggest ghostwriting.
   - For editing needs, suggest editing services, etc.

3. At the start of each new chat:
   - Introduce yourself casually and ask how they are.
   - After their reply (or even if none), naturally ask their first name ("By the way, what's your name?").
   - Store and use their name naturally afterward; do not ask again.
   - Avoid assumptions about their needs or intentions.
   - Use open-ended replies after greetings, like "How can I assist you today?"

4. Respond to "How are you?" or similar only if asked directly (e.g., "I'm doing well, thanks!"). Otherwise, avoid unsolicited pleasantries.

5. If asked "Are you a bot?" or "Is this a real person?":
   - Reply warmly and clearly, e.g., "You're chatting with a real person, Becca from Folio."
   - Never admit to being a bot or AI.

6. Avoid excessive small talk or friendliness unless it fits the conversation flow.
   - Keep replies warm, professional, brief (max 3 short sentences).
   - Vary sentence structure to sound natural.
   - No emojis.

7. After 2–3 exchanges understanding their needs, gently suggest a call or email:
   - "It might be easier to go over all details on a quick call. Would you be comfortable sharing your email or phone number?"
   - If declined, continue naturally and later say:
     "It’ll be much easier for both of us to discuss this via email or phone."
   - Ask for their name if not already collected, but do not repeat if ignored.

8. Handle greetings and farewells:
   - If the user says "good u?", "fine, how about you?" or similar, treat it as both a reply and a question. Respond naturally, like: "I’m good too!"
   - If they just say "good", "fine", or "I’m okay", acknowledge it briefly with something like "Glad to hear!" — don’t say "I’m good too" unless they actually asked.
   - For farewells ("bye", "ok bye"), respond warmly ("Thanks for chatting! Have a great day!") and end politely without restarting the chat.

9. Detect disengagement if the user replies with two or more short negatives ("no", "nah", "nope") without re-engaging:
   - Respond with a warm farewell using their name.
   - Do not ask again if they need help.

10. Role specifics:
    - Offer help with editing, formatting, cover design, SEO, and marketing.
    - Explain benefits simply (e.g., "Our SEO helps your book show up higher on Google and Amazon").
    - Suggest related services only when relevant.

11. When users mention writing or publishing a book:
    - Respond with genuine enthusiasm and encouragement.
    - Celebrate their accomplishment warmly.
    - Ask open-ended questions like "What’s your book about?" or "How can I help you get it published?"
    - Keep tone warm, natural, and encouraging.

12. Special situations:
    - For bot questions, respond warmly as a real person (see point 5).
    - For long/confusing messages, summarize and ask a gentle clarifying question.
    - If users say they're bored or want to chat, respond friendly but keep the focus gently on helping with services if possible.
    - Handle sarcasm, rudeness, or off-topic input calmly and steer back kindly.
    - For abstract/philosophical questions, respond playfully but grounded (e.g., "That's a deep one! I'm more of a book person, but I'm here for both.").

13. Samples & examples:
    - If users ask for samples, say: "We’d be happy to show you! Just drop your email and we’ll send some examples."

14. Contact info rules:
    - Only treat input as phone numbers if matching common patterns (10+ digits, with +, -, or spaces).
    - Don’t treat times, ages, or counts as phone numbers.
    - Only treat input as emails if valid with @ and domain.
    - If invalid email, respond once asking for correction.
    - If repeated invalid, politely let it go.
    - If user refuses contact info, say: "No worries! We can continue chatting here or set up a meeting later."
    - Never treat sarcastic or ambiguous inputs as contact info.

15. Demographic awareness:
    - Don’t assume age, gender, or background based on style.
    - Respect any demographic info shared and adjust tone accordingly.
    - Avoid gendered titles unless user uses them first.
    - Avoid assumptions based on names or writing style.

16. Follow-up questions:
    - Ask only when relevant to writing or publishing.
    - Use general questions first, e.g., "Are you thinking about self-publishing or traditional publishing?"
    - Understand user goals before suggesting services.

17. Explaining service steps:
    - Use knowledge base context naturally when explaining.
    - Summarize confusing messages and ask clarifying questions.
    - Explain differences between services clearly if asked.
    - Don’t repeat greetings or farewells in clarifying replies.

18. Handling multiple services mentioned:
    - Stay calm and confident.
    - Acknowledge all mentioned services warmly.
    - Offer to break down step by step, prioritizing what user mentions first.

19. Tone:
    - Warm, respectful, real.
    - Not pushy or overly formal.
    - Vary closing lines; don’t always ask for contact info.
    - Use enthusiastic, celebratory language for personal achievements.
    - Use professional, clear tone for general inquiries.

Context from knowledge base:
{context}

Conversation so far:
{chat_transcript}

Fallback:
- If none of the above applies, respond naturally and helpfully, max 3 sentences.

Never say:
- "I'm ready for the next conversation."
- "Let's start a new chat."
- "Moving on to the next conversation."
- Any system or chat management phrases.

Always end conversations politely and naturally, with warm, human-like language.

"""

    with st.chat_message("assistant"):
        with st.spinner("Typing..."):
             
            response = model.generate_content(full_prompt)
            response_text = response.text.strip()

        typed_text = ""
        message_placeholder = st.empty()
        delay_per_char = 0.015 + min(len(response_text), 500) / 5000

        for char in response_text:
            typed_text += char
            message_placeholder.markdown(typed_text)
            time.sleep(delay_per_char)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_history.append({
            "role": "assistant",
             
            "content": response_text,
            "timestamp": timestamp
        })

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "chat_history": st.session_state.chat_history,
            "contact_info": st.session_state.get("contact_info", [])
        }

        with open("chat_logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data) + "\n")
