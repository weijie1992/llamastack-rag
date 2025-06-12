import gradio as gr
from llama_stack_client import LlamaStackClient
import json
from typing import List, Tuple, Dict, Any
from llama_stack_client.types import SamplingParams


class RAGChatbot:
    def __init__(
        self, llama_stack_port: str = "8321", vector_db_id: str = "service_requests_db"
    ):
        """Initialize the RAG chatbot with LlamaStack client."""
        self.client = LlamaStackClient(base_url=f"http://localhost:{llama_stack_port}")
        self.vector_db_id = vector_db_id
        self.model_id = "meta-llama/Llama-3.2-1B-Instruct"  # Adjust based on your setup

        # Test the connection and print available methods
        self.test_connection()

    def test_connection(self):
        """Test the connection and inspect available methods."""
        try:
            print("Testing LlamaStack connection...")
            print(f"Client base URL: {self.client.base_url}")

            # Check vector_io methods
            if hasattr(self.client, "vector_io"):
                print("vector_io methods available:")
                vector_io_methods = [
                    method
                    for method in dir(self.client.vector_io)
                    if not method.startswith("_")
                ]
                print(f"  {vector_io_methods}")

            # Try to list available vector databases
            try:
                # This might help identify the correct method
                dbs = self.client.vector_io.list_vector_dbs()
                print(f"Available vector databases: {dbs}")
            except Exception as e:
                print(f"Could not list vector databases: {e}")

        except Exception as e:
            print(f"Connection test failed: {e}")

    def retrieve_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from vector database."""
        try:
            # Use the correct API format with params dictionary
            vector_db_response = self.client.vector_io.query(
                vector_db_id=self.vector_db_id,
                query=query,
                params={
                    "k": k,  # top_k parameter goes in params dict
                    "include_metadata": True,  # Include metadata if available
                },
            )

            print(f"Vector DB Response Type: {type(vector_db_response)}")
            print(f"Vector DB Response: {vector_db_response}")

            # QueryChunksResponse has chunks: List[Chunk] and scores: List[float]
            documents = []

            if hasattr(vector_db_response, "chunks") and vector_db_response.chunks:
                chunks = vector_db_response.chunks
                scores = getattr(vector_db_response, "scores", [])

                print(f"Found {len(chunks)} chunks with {len(scores)} scores")

                # Combine chunks with their scores
                for i, chunk in enumerate(chunks):
                    score = scores[i] if i < len(scores) else None

                    # Create a dict combining chunk data with score
                    chunk_dict = {"chunk": chunk, "score": score, "index": i}
                    documents.append(chunk_dict)

                print(f"First chunk type: {type(chunks[0])}")
                if hasattr(chunks[0], "__dict__"):
                    print(
                        f"First chunk attributes: {[attr for attr in dir(chunks[0]) if not attr.startswith('_')]}"
                    )

                # Show first chunk content for debugging
                print(f"First chunk sample: {str(chunks[0])[:200]}...")

            else:
                print(
                    f"Response attributes: {[attr for attr in dir(vector_db_response) if not attr.startswith('_')]}"
                )
                print("No chunks found in response")

            return documents

        except Exception as e:
            print(f"Error retrieving from vector DB: {e}")
            print(f"Exception type: {type(e)}")

            # Try without params as fallback
            try:
                print("Trying query without params...")
                vector_db_response = self.client.vector_io.query(
                    vector_db_id=self.vector_db_id, query=query
                )
                print(f"Fallback query response: {vector_db_response}")

                if hasattr(vector_db_response, "chunks"):
                    chunks = (
                        vector_db_response.chunks[:k]
                        if vector_db_response.chunks
                        else []
                    )
                    scores = getattr(vector_db_response, "scores", [])[:k]

                    documents = []
                    for i, chunk in enumerate(chunks):
                        score = scores[i] if i < len(scores) else None
                        documents.append({"chunk": chunk, "score": score, "index": i})
                    return documents

            except Exception as e2:
                print(f"Fallback query also failed: {e2}")

            return []

    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string."""
        if not retrieved_docs:
            return ""

        context_parts = []
        for i, doc_dict in enumerate(retrieved_docs, 1):
            chunk = doc_dict.get("chunk")
            score = doc_dict.get("score", "N/A")

            content = ""

            # Handle different chunk structures
            if hasattr(chunk, "content"):
                content = chunk.content
            elif hasattr(chunk, "text"):
                content = chunk.text
            elif hasattr(chunk, "document"):
                content = chunk.document
            elif hasattr(chunk, "data"):
                content = chunk.data
            elif isinstance(chunk, dict):
                content = chunk.get(
                    "content", chunk.get("text", chunk.get("document", str(chunk)))
                )
            else:
                content = str(chunk)

            # Format with score
            score_str = (
                f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            )
            context_parts.append(f"Document {i} (Score: {score_str}):\n{content}")

        print(f"==>> context_parts: {context_parts}")
        return "\n\n".join(context_parts)

    def create_rag_prompt(self, query: str, context: str) -> str:
        """Create a RAG prompt combining context and user query."""
        if not context:
            return f"""You are a helpful AI assistant. Please answer the following question:
                Question: {query}
                Answer:"""

        return f"""You are a helpful AI assistant. Use the provided context to answer the user's question. If the context doesn't contain relevant information, say so and provide a general response.
            Context:
            {context}
            Question: {query}
            Answer:"""

    def generate_response(self, prompt: str) -> str:
        """Generate response using the LLM."""
        print(f"==>> generate_response:")

        try:
            response = self.client.inference.completion(
                model_id=self.model_id,
                content=prompt,
                sampling_params={"max_tokens": 512, "temperature": 0.7, "top_p": 0.9},
                # sampling_params={
                #     "strategy": {
                #         "type": "greedy",
                #     },
                #     "max_tokens": 50,
                # },
            )
            print(f"==>> response: {response}")

            if hasattr(response, "content") and response.content:
                return (
                    response.content[0].text
                    if isinstance(response.content, list)
                    else response.content
                )
            elif hasattr(response, "text"):
                return response.text
            elif hasattr(response, "completion"):
                return response.completion
            else:
                return str(response)

        except Exception as e:
            print(f"Error generating response: {e}")
            # Fallback to chat completion if inference.completion fails
            try:
                messages = [{"role": "user", "content": prompt}]
                response = self.client.inference.chat_completion(
                    model_id=self.model_id,
                    messages=messages,
                    sampling_params={
                        "max_tokens": 512,
                        "temperature": 0.7,
                        "top_p": 0.9,
                    },
                )

                if hasattr(response, "completion_message"):
                    return response.completion_message.content
                else:
                    return str(response)

            except Exception as e2:
                print(f"Error with chat completion: {e2}")
                return f"Sorry, I encountered an error generating a response: {str(e2)}"

    def rag_chat(
        self, message: str, history: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], str]:
        """Main RAG chat function."""
        print(f"User message: {message}")
        print(f"Chat history length: {len(history)}")

        # Step 1: Retrieve relevant context
        retrieved_docs = self.retrieve_context(message)
        print(f"Retrieved {len(retrieved_docs)} documents")

        # Step 2: Format context
        context = self.format_context(retrieved_docs)
        if context:
            print(f"Context length: {len(context)} characters")
        else:
            print("No context retrieved")

        # Step 3: Create RAG prompt
        rag_prompt = self.create_rag_prompt(message, context)
        print(f"==>> rag_prompt: {rag_prompt}")

        # Step 4: Generate response
        response = self.generate_response(rag_prompt)

        print(f"Generated response: {response[:100]}...")

        # Update history
        history.append((message, response))

        return history, ""


# Initialize RAG chatbot
rag_bot = RAGChatbot()


def chatbot_interface(message, history):
    """Interface function for Gradio."""
    return rag_bot.rag_chat(message, history)


def clear_chat():
    """Clear chat history."""
    return [], ""


# Create Gradio interface
with gr.Blocks(title="RAG Chatbot with LlamaStack") as demo:
    gr.Markdown("# 🤖 RAG Chatbot with LlamaStack")
    gr.Markdown(
        "Ask questions and I'll search through the knowledge base to provide relevant answers!"
    )

    # Display current configuration
    with gr.Accordion("Configuration", open=False):
        gr.Markdown(
            f"""
        **Current Setup:**
        - Vector DB ID: `{rag_bot.vector_db_id}`
        - Model ID: `{rag_bot.model_id}`
        - LlamaStack URL: `{rag_bot.client.base_url}`
        """
        )

    chatbot = gr.Chatbot(value=[], elem_id="chatbot", height=500, show_copy_button=True)

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask a question about your documents...",
            container=False,
            scale=7,
            label="Your Question",
        )
        submit = gr.Button("Send", scale=1, variant="primary")
        clear = gr.Button("Clear Chat", scale=1)

    # Event handlers
    submit.click(chatbot_interface, inputs=[msg, chatbot], outputs=[chatbot, msg])

    msg.submit(chatbot_interface, inputs=[msg, chatbot], outputs=[chatbot, msg])

    clear.click(clear_chat, outputs=[chatbot, msg])

# Launch the application
if __name__ == "__main__":
    print("Starting RAG Chatbot...")
    print(f"Vector DB ID: {rag_bot.vector_db_id}")
    print(f"Model ID: {rag_bot.model_id}")

    demo.launch(
        share=False,  # Set to True for public sharing
        server_port=7860,
        server_name="0.0.0.0",
    )
