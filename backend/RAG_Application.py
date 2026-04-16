

import gradio as gr
import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import LLMChain, PromptTemplate
from langchain_huggingface import HuggingFacePipeline
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
import os
from openai import OpenAI
import anthropic
from gtts import gTTS
import tempfile
import pygame
from pathlib import Path
from langchain.llms import Ollama

# Device setup if MacOS
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print('mps is used.')
else:
    device = 'cpu'
    print('cpu is used.')

# Device Setup if Windows with Nvidia Windows

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Anthropic API setup
anthropic_client = anthropic.Client(api_key="PASTE YOUR ANTHROPIC")

# OpenAI API setup
client_openai = OpenAI(api_key = "PASTE YOUR OPENAI API KEY")

#Connecting Ollama to the Langchain
llm = Ollama(
    model="llama3.2:latest",  # Use the exact model name from `ollama list`
    base_url="http://127.0.0.1:11434",  # Default Ollama server URL ######PASTE YOUR OLLAMA URL 
)
pygame.mixer.init() #comment out if no speakers 

# Text to Speech
def text_to_speech(text: str) -> str:
    """Convert text to speech and play it."""
    try:
        # Create a temporary directory if it doesn't exist
        temp_dir = Path("temp_audio")
        temp_dir.mkdir(exist_ok=True)
        
        # Create temporary file for audio
        temp_file = temp_dir / "temp_speech.mp3"
        
        # Generate speech
        tts = gTTS(text=text, lang='en')
        tts.save(str(temp_file))
        
        # Play the audio
        pygame.mixer.music.load(str(temp_file))
        pygame.mixer.music.play()
        
        return "Playing audio..."
    except Exception as e:
        return f"Error generating audio: {str(e)}"
    

def normalize_for_comparison(text: str) -> str:
    """
    Normalize text for comparison by removing all whitespace and punctuation.
    """
    # Remove all whitespace, punctuation, and convert to lowercase
    return ''.join(char.lower() for char in text if not char.isspace() and char.isalnum())


# Query the Chroma
def query_chroma(query: str, collection: chromadb.Collection, top_k: int = 3) -> List[Dict[str, any]]:
    """
    Query the Chroma database and return top_k in this case 3, unique most relevant results.
    Handles cases where texts might be formatted differently but contain the same content.
    Specifically handles whitespace variations and minor formatting differences.
    """
    # Query with more results than needed to account for duplicates
    results = collection.query(
        query_texts=[query],
        n_results=top_k * 5  # Request more results to ensure enough unique entries after filtering
    )
    
    # Create a list to track processed results
    formatted_results = []
    # Track normalized versions of added texts
    seen_normalized_texts = set()
    
    # Process results and remove similar entries
    for i in range(len(results['ids'][0])):
        current_text = results['documents'][0][i]
        normalized_text = normalize_for_comparison(current_text)
        
        # Skip if we've seen this normalized text before
        if normalized_text not in seen_normalized_texts:
            seen_normalized_texts.add(normalized_text)
            formatted_results.append({
                'id': results['ids'][0][i],
                'text': current_text,  # Keep original formatting for display
                'distance': results['distances'][0][i]
            })
            
        # Break if we have enough unique results
        if len(formatted_results) == top_k:
            break
            
    return formatted_results

def format_context(results: List[Dict[str, any]]) -> str:
    """Format the ChromaDB results into a string context."""
    context_parts = []
    for i, result in enumerate(results, 1):
        context_parts.append(f"{i}. {result['text']}")
    return "\n\n".join(context_parts)

# Setup of Ollama and call to local LLM and initlizing of chromadb client
# Create prompt template
standard_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert AI assistant specializing in haptics domain, tasked with providing accurate, concise responses.
Explain concepts using the context like i am a 15 year old.
Offer clear, direct explanations of key haptic concepts.
You are instructed to use the context given below to answer the user question.
If the context is not of good quality then provide a general answer to the user question.
The context is  you need to derive your answer from = 
{context}
The question is =
<|eot_id|><|start_header_id|>user<|end_header_id|>
{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>Your Answer = """

standard_prompt = PromptTemplate(template=standard_template, input_variables=["query", "context"])
standard_chain = LLMChain(prompt=standard_prompt, llm=llm, verbose=False)

# Initialize ChromaDB client
client = chromadb.PersistentClient(
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE
)
collection = client.get_collection(name="research_papers")

# Clean the response from the Local LLM so that we can separate out the context, prompt and the generated answer for user clarity.
def clean_response(response: str) -> str:
    """Clean the response to show only the generated answer."""
    # First find everything after "The question is ="
    if "The question is =" in response:
        response = response.split("The question is =")[1]
        
    # Remove any remaining system markers
    markers = [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "system",
        "user",
        "assistant",
        "Your Answer ="
    ]
    
    cleaned = response
    for marker in markers:
        cleaned = cleaned.replace(marker, "")
    
    # Remove any lines containing unwanted phrases
    cleaned = "\n".join([line for line in cleaned.split('\n') 
    if not any(x in line.lower() for x in ["date:", "cutting", "today"])])
    
    # Final cleanup
    cleaned = cleaned.strip()
    
    return cleaned

# Enhanced RAG Approach
def enchanced_rag(message: str) -> str:
    """Enhanced RAG approach: LLM generating small rewriting before vector search."""

    enhance_rag_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a haptics domain researcher instructed to answer the user question in max 3 lines in the tone of a research paper. Your task is to write a small answer to the given question with respect to Haptics. It must be concise 2-line answer that captures the core information need. Focus on the key technical concepts and terminology.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    enhance_rag_prompt = PromptTemplate(template=enhance_rag_template, input_variables=["query"])
    enhance_rag_chain = LLMChain(prompt=enhance_rag_prompt, llm=llm, verbose=False)

    try:
        llm_generated_reference_answer = enhance_rag_chain.run(query=message)
        print(f"llm_generated_reference_answer: {llm_generated_reference_answer}")
        
        results = query_chroma(llm_generated_reference_answer, collection)
        context = format_context(results)
        
        response = standard_chain.run(query=message, context=context)
        return clean_response(response)
    except Exception as e:
        return f"An error occurred: {str(e)}"


# API Call to Claude.
def claude_api(message: str) -> str:
    """Direct API call to Claude."""
    try:
        message = f"""You are a haptics domain expert. Please answer the following question:
        
Question: {message}

Please provide a clear and concise response focusing on haptics-related information."""

        response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            temperature=0.7,
            system="""You are an AI assistant tasked with answering questions related to the domain of haptics. Your goal is to provide informative, accurate, and concise responses that demonstrate a deep understanding of haptic technology and its applications.
When answering the question, please follow these guidelines:
- Provide a clear and concise explanation of the relevant haptic concepts
- Use technical terms related to haptics, but briefly explain them if they are not commonly known
- If the question is broad, try to cover multiple aspects of the topic within the word limit

Please structure your answer as follows:
1. A brief introduction to the topic
2. Main body of the answer, addressing key points related to the question
3. A concise conclusion or summary

Your answer should be between 300 and 400 words. Make sure to stay focused on the domain of haptics and related technologies. 

Please provide your answer within <answer> tags. Begin your response now:
<answer>""",
            messages=[
                {"role": "user", "content": message}
            ]
        )
        return response.content[0].text
    except Exception as e:
        return f"An error occurred while calling Claude API: {str(e)}"

# API call to OpenAI
def gpt_api(message: str) -> str:
    """Direct API call to OpenAI's GPT model."""
    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """You are an AI assistant tasked with answering questions related to the domain of haptics. Your goal is to provide informative, accurate, and concise responses that demonstrate a deep understanding of haptic technology and its applications.

When answering the question, please follow these guidelines:
- Provide a clear and concise explanation of the relevant haptic concepts
- Use technical terms related to haptics, but briefly explain them if they are not commonly known
- If the question is broad, try to cover multiple aspects of the topic within the word limit

Please structure your answer as follows:
1. A brief introduction to the topic
2. Main body of the answer, addressing key points related to the question
3. A concise conclusion or summary

Your answer should be between 300 and 400 words. Make sure to stay focused on the domain of haptics and related technologies. If the question is not directly related to haptics, try to connect it to haptic concepts or applications if possible.

Please provide your answer within <answer> tags. Begin your response now:
<answer>"""},
                {"role": "user", "content": message}
            ]
        )
        return response.choices[0].message.content if response.choices else 'No content generated.'
    except Exception as e:
        return f"An error occurred while calling GPT API: {str(e)}"
    
    
# User query call to the Chromadb, and sending the query to all the three models 
def process_query(message: str) -> tuple[str, str, str, str]:
    """Process the query and return context and responses from all models."""
    try:
        # Get context and RAG response
        results = query_chroma(message, collection)
        context = format_context(results)
        rag_response = standard_chain.run(query=message, context=context)
        # enchanced_rag_reponse = enchanced_rag(message)   #this is the enhanced rag approach, we didn't implement it in the user study prototype as its resource intensive
        cleaned_rag_response = clean_response(rag_response)
        # print("-----------------------------------\n",enchanced_rag_reponse)
        # Get Claude API response
        claude_response = claude_api(message)
        
        # Get GPT API response
        gpt_response = gpt_api(message)
        
        return context, cleaned_rag_response, claude_response, gpt_response
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return error_message, error_message, error_message, error_message

def process_query_and_clear(message: str) -> tuple[str, str, str, str, str]:
    """Process the query and return context, responses from all models, and empty string to clear input."""
    context, rag_response, claude_response, gpt_response = process_query(message)
    return context, rag_response, claude_response, gpt_response, ""


"""
Following section is the creating of web UI using gradio library
"""
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# Llama 3.2 Haptics QA System")
    gr.Markdown("Ask questions about Haptics and compare responses from different models.")
    with gr.Row():
        input_text = gr.Textbox(
            label="Enter your question:",
            placeholder="Type your haptics-related question here...",
            lines=2
        )
        submit_btn = gr.Button("Send", variant="primary", size = "sm")
        clear_btn = gr.Button("Clear", variant="secondary",size = "sm")

    with gr.Row():
        with gr.Column():
            with gr.Group():
                context_box = gr.Textbox(
                    label="Retrieved Context",
                    lines=20,
                    interactive=False,
                    visible=True,
                    max_lines=20,
                )
                context_audio_btn = gr.Button("🔊 Play (Disabled for User Study)", size="sm")
                context_audio_status = gr.Textbox(label="Audio Status", interactive=False, visible=False)
        
        with gr.Column():
            with gr.Group():
                rag_response_box = gr.Textbox(
                    label="RAG Response",
                    lines=20,
                    interactive=False,
                    visible=True,
                    max_lines=20,
                )
                rag_audio_btn = gr.Button("🔊 Play (Disabled for User Study)", size="sm")
                rag_audio_status = gr.Textbox(label="Audio Status", interactive=False, visible=False)
    
    with gr.Row():
        with gr.Column():
            with gr.Group():
                claude_response_box = gr.Textbox(
                    label="Claude API Response",
                    lines=20,
                    interactive=False,
                    visible=True,
                    max_lines=20,
                )
                claude_audio_btn = gr.Button("🔊 Play (Disabled for User Study)", size="sm")
                claude_audio_status = gr.Textbox(label="Audio Status", interactive=False, visible=False)
        
        with gr.Column():
            with gr.Group():
                gpt_response_box = gr.Textbox(
                    label="OpenAI API Response",
                    lines=20,
                    interactive=False,
                    visible=True,
                    max_lines=20,
                            )
                gpt_audio_btn = gr.Button("🔊 Play (Disabled for User Study)", size="sm")
                gpt_audio_status = gr.Textbox(label="Audio Status", interactive=False, visible=False)

    # Setup click handlers for audio buttons
    context_audio_btn.click(
        text_to_speech,
        inputs=[context_box],
        outputs=[context_audio_status]
    )
    
    rag_audio_btn.click(
        text_to_speech,
        inputs=[rag_response_box],
        outputs=[rag_audio_status]
    )
    
    claude_audio_btn.click(
        text_to_speech,
        inputs=[claude_response_box],
        outputs=[claude_audio_status]
    )
    
    gpt_audio_btn.click(
        text_to_speech,
        inputs=[gpt_response_box],
        outputs=[gpt_audio_status]
    )

    # Modified submission handlers to clear input
    input_text.submit(
        process_query_and_clear,
        inputs=[input_text],
        outputs=[context_box, rag_response_box, claude_response_box, gpt_response_box, input_text]
    )
    # input_text.submit(
    #     process_query,
    #     inputs=[input_text],
    #     outputs=[context_box, rag_response_box, claude_response_box, gpt_response_box]
    # )
    submit_btn.click(
        process_query_and_clear,
        inputs=[input_text],
        outputs=[context_box, rag_response_box, claude_response_box, gpt_response_box]
    )
    clear_btn.click(lambda: "", inputs=[], outputs=[input_text])


# Cleanup function for temporary audio files
def cleanup_temp_audio():
    temp_dir = Path("temp_audio")
    if temp_dir.exists():
        for file in temp_dir.glob("*.mp3"):
            try:
                file.unlink()
            except:
                pass
        try:
            temp_dir.rmdir()
        except:
            pass

# Register cleanup function
import atexit
atexit.register(cleanup_temp_audio)

if __name__ == "__main__":
    demo.launch(share=True)