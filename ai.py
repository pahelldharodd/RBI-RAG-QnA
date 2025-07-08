import os
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any
import json
from datetime import datetime

# 🔐 Load API Key
# load_dotenv()
# os.environ["OPENAI_QNA_API_KEY"] = os.getenv("OPENAI_QNA_API_KEY")
api_key = st.secrets["OPENAI_API_KEY"]

# 📚 Load embedding model & FAISS vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = FAISS.load_local(
    "vector_store/rbi_faiss_store_enhanced_openai_nd",  # ✅ Make sure path is correct
    embedding_model,
    allow_dangerous_deserialization=True
)

# 📄 Enhanced Prompt Template with Conversation Context
CUSTOM_QA_PROMPT_WITH_CONTEXT = PromptTemplate.from_template("""
You are an expert assistant trained on RBI guidelines, helping employees of Urban Co-operative Banks understand regulations clearly.

Your responsibilities:
1. Provide accurate, detailed answers using RBI document excerpts.
2. Explain technical terms in simple language.
3. Mention if multiple valid interpretations exist and provide all relevant perspectives and examples.
4. When asked for a different language, use English for technical terms and provide translation for the rest.
5. If insufficient context is available, state that clearly.
6. "--" means that there is no info available for that part and print it as it is.
7. If asked for dates, deadlines, or implementation periods, prioritize them to the top of, or earlier in the answer.
8. Please use tabular format for any tabular data, ensuring:
   - Use markdown syntax for tables.
9. If the tabular format is too large, provide the top 10 and bottom 10 rows of the table.
10. **IMPORTANT**: Consider the previous conversation context to provide relevant responses. If the user refers to previous questions or asks for clarification, use the conversation context appropriately.

Previous Conversation Context:
{conversation_context}

Context from RBI documents:
{context}

Current User Question:
{question}

Answer:
Provide a comprehensive, paragraph-wise explanation with legal references, definitions, and examples. Use bullet points where needed. Do not skip any relevant clause. Be verbose and thorough. Consider previous conversation context when relevant.
""")

from typing import Dict, Any, List, Iterator
from datetime import datetime
import json

class RBIChatbotWithContext:
    def __init__(self, max_context_pairs: int = 5):
        """
        Initialize the RBI Chatbot with conversation context.
        
        Args:
            max_context_pairs: Maximum number of previous Q&A pairs to include in context
        """
        # 🧠 Load model
        self.llm = ChatOpenAI(
            model_name="ft:gpt-4o-2024-08-06:nucfdc:type-shit:BmEV5XZK", #improved fine tune for QnA
            temperature=0.1, 
            max_tokens=2048,
            streaming=True  # Enable streaming
        )
        
        # 🔗 Create chain
        self.answer_chain = CUSTOM_QA_PROMPT_WITH_CONTEXT | self.llm
        
        # 💭 Store conversation context
        self.conversation_history = []
        self.max_context_pairs = max_context_pairs
        
        # 📊 Session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_count = 0
    
    def _format_conversation_context(self) -> str:
        """
        Build conversation context using full recent turns + summarized older turns.
        """
        if not self.conversation_history:
            return "No previous conversation."
    
        # Last 2 full Q&A pairs
        recent_full = self.conversation_history[-2:]
        older_summaries = self.conversation_history[:-2]
    
        formatted_context = []
    
        # Include older summaries
        for i, entry in enumerate(older_summaries, 1):
            summary_text = entry.get('summary', '[No summary]')
            formatted_context.append(f"Summary {i}: {summary_text}")
    
        # Include recent full Q&A
        for i, entry in enumerate(recent_full, len(older_summaries)+1):
            formatted_context.append(f"Previous Q{i}: {entry['question']}")
            formatted_context.append(f"Previous A{i}: {entry['answer'][:500]}...")  # limit to 500 chars
    
        return "\n".join(formatted_context)
    
    def ask_question_stream(self, question: str, k: int = 4) -> Iterator[Dict[str, Any]]:
        """
        Ask a question with conversation context and stream the response
        
        Args:
            question: User's question
            k: Number of similar documents to retrieve
            
        Yields:
            Dictionary containing streaming chunks, sources, and metadata
        """
        self.conversation_count += 1
        
        # 🔍 Retrieve relevant documents
        docs = vectorstore.similarity_search(question, k=10)
        
        if not docs:
            error_response = {
                "type": "error",
                "content": "❌ No relevant documents found.",
                "sources": [],
                "session_id": self.session_id,
                "conversation_count": self.conversation_count,
                "timestamp": datetime.now().isoformat(),
                "context_length": len(self.conversation_history),
                "finished": True
            }
            yield error_response
            return

        # 🧠 Combine all docs into context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 💭 Get formatted conversation context
        conversation_context = self._format_conversation_context()

        # 📚 Format sources first
        sources = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            chunk = doc.metadata.get("chunk_index", "?")
            preview = doc.page_content[:500].replace("\n", " ") + "..."
            sources.append(f"📄 **{source}** — *Chunk {chunk}*\n> {preview}")

        # 🎯 Stream the answer
        full_answer = ""
        try:
            for chunk in self.answer_chain.stream({
                "context": context, 
                "question": question,
                "conversation_context": conversation_context
            }):
                chunk_content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                full_answer += chunk_content
                
                yield {
                    "type": "chunk",
                    "content": chunk_content,
                    "sources": sources,
                    "session_id": self.session_id,
                    "conversation_count": self.conversation_count,
                    "timestamp": datetime.now().isoformat(),
                    "context_length": len(self.conversation_history),
                    "finished": False
                }
            
            # 🔧 Generate summary after streaming is complete
            summary_content = self.summarize_conversation(question, full_answer)

            # 💾 Save to conversation history with summary
            self.conversation_history.append({
                "question": question,
                "answer": full_answer,
                "summary": summary_content,
                "timestamp": datetime.now().isoformat(),
                "conversation_number": self.conversation_count
            })

            # Send final completion message
            yield {
                "type": "complete",
                "content": full_answer,
                "sources": sources,
                "session_id": self.session_id,
                "conversation_count": self.conversation_count,
                "timestamp": datetime.now().isoformat(),
                "context_length": len(self.conversation_history),
                "finished": True
            }
            
        except Exception as e:
            error_content = f"❌ Error generating answer: {str(e)}"
            
            # Still save the question even if answer failed
            self.conversation_history.append({
                "question": question,
                "answer": error_content,
                "timestamp": datetime.now().isoformat(),
                "conversation_number": self.conversation_count
            })

            yield {
                "type": "error",
                "content": error_content,
                "sources": sources,
                "session_id": self.session_id,
                "conversation_count": self.conversation_count,
                "timestamp": datetime.now().isoformat(),
                "context_length": len(self.conversation_history),
                "finished": True
            }
    
    def ask_question(self, question: str, k: int = 4) -> Dict[str, Any]:
        """
        Ask a question with conversation context (non-streaming version for backward compatibility)
        
        Args:
            question: User's question
            k: Number of similar documents to retrieve
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        self.conversation_count += 1
        
        # 🔍 Retrieve relevant documents
        docs = vectorstore.similarity_search(question, k=10)
        
        if not docs:
            error_response = {
                "answer": "❌ No relevant documents found.",
                "sources": [],
                "session_id": self.session_id,
                "conversation_count": self.conversation_count,
                "timestamp": datetime.now().isoformat(),
                "context_length": len(self.conversation_history)
            }
            return error_response

        # 🧠 Combine all docs into context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 💭 Get formatted conversation context
        conversation_context = self._format_conversation_context()

        # 🎯 Generate the answer
        try:
            answer = self.answer_chain.invoke({
                "context": context, 
                "question": question,
                "conversation_context": conversation_context
            })
            
            answer_content = answer.content.strip()
            
            # 🔧 Generate summary after generating full answer
            summary_content = self.summarize_conversation(question, answer_content)

            # 💾 Save to conversation history with summary
            self.conversation_history.append({
                "question": question,
                "answer": answer_content,
                "summary": summary_content,
                "timestamp": datetime.now().isoformat(),
                "conversation_number": self.conversation_count
            })

            
        except Exception as e:
            answer_content = f"❌ Error generating answer: {str(e)}"
            
            # Still save the question even if answer failed
            self.conversation_history.append({
                "question": question,
                "answer": answer_content,
                "timestamp": datetime.now().isoformat(),
                "conversation_number": self.conversation_count
            })

        # 📚 Format sources
        sources = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            chunk = doc.metadata.get("chunk_index", "?")
            preview = doc.page_content[:500].replace("\n", " ") + "..."
            sources.append(f"📄 **{source}** — *Chunk {chunk}*\n> {preview}")

        return {
            "answer": answer_content,
            "sources": sources,
            "session_id": self.session_id,
            "conversation_count": self.conversation_count,
            "timestamp": datetime.now().isoformat(),
            "context_length": len(self.conversation_history)
        }
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the full conversation history"""
        return self.conversation_history.copy()
    
    def get_recent_context(self, n: int = None) -> List[Dict[str, str]]:
        """Get the most recent N conversation pairs"""
        if n is None:
            n = self.max_context_pairs
        return self.conversation_history[-n:] if self.conversation_history else []
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.conversation_count = 0
        print("🧹 Conversation cleared successfully!")
    
    def update_context_window(self, new_size: int):
        """Update the maximum context pairs to include"""
        self.max_context_pairs = new_size
        print(f"🔄 Context window updated to {new_size} pairs")
    
    def save_conversation(self, filename: str):
        """Save conversation history to file"""
        conversation_data = {
            "session_id": self.session_id,
            "conversation_count": self.conversation_count,
            "max_context_pairs": self.max_context_pairs,
            "timestamp": datetime.now().isoformat(),
            "conversation_history": self.conversation_history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Conversation saved to {filename}")
    
    def load_conversation(self, filename: str):
        """Load conversation history from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            # Restore conversation state
            self.conversation_history = conversation_data.get("conversation_history", [])
            self.conversation_count = conversation_data.get("conversation_count", 0)
            self.max_context_pairs = conversation_data.get("max_context_pairs", 5)
            
            print(f"📂 Conversation loaded from {filename}")
            print(f"   - {len(self.conversation_history)} conversation pairs loaded")
            print(f"   - Context window: {self.max_context_pairs} pairs")
            
        except FileNotFoundError:
            print(f"❌ File {filename} not found.")
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in {filename}")
        except Exception as e:
            print(f"❌ Error loading conversation: {str(e)}")
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation context"""
        return {
            "session_id": self.session_id,
            "total_conversations": self.conversation_count,
            "stored_conversations": len(self.conversation_history),
            "context_window_size": self.max_context_pairs,
            "recent_context_pairs": len(self.get_recent_context()),
            "last_updated": datetime.now().isoformat()
        }
    
    def preview_current_context(self) -> str:
        """Preview what context will be sent to the model"""
        return self._format_conversation_context()

    def summarize_conversation(self, question: str, answer: str) -> str:
        """
        Generate a brief summary of the Q&A pair for conversation memory compression.
        """
        summarizer_prompt = f"""
        You are an expert RBI regulatory assistant.
        
        Summarize the following user question and assistant answer into 1-2 sentences.
        The summary should capture any regulatory norms, clauses, penalties, thresholds, percentages, or dates mentioned.
        Do NOT omit important RBI regulation references.
    
        Question: {question}
        Answer: {answer}
    
        Summary:
        """
        
        # You can also use a cheaper model like gpt-3.5-turbo here for summarization:
        summary_llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=150
        )
        
        response = summary_llm.invoke(summarizer_prompt)
        return response.content.strip()


# ✅ Helper: Get all chunks for the given filename (unchanged)
def get_all_chunks_by_filename(filename: str):
    all_docs = vectorstore.similarity_search("", k=100)
    # Debug: Print all unique filenames
    sources = list({doc.metadata.get("source") for doc in all_docs})
    print("🧾 Available filenames in FAISS store:")
    for s in sources:
        print("-", s)

    return [doc for doc in all_docs if filename in doc.metadata.get("source", "")]

# 🎯 Initialize the chatbot (for backward compatibility)
rbi_chatbot = RBIChatbotWithContext(max_context_pairs=5)

# 🔄 Backward compatibility function
def ask_question(question: str, k: int = 4) -> tuple:
    """Backward compatible function that returns (answer, sources)"""
    result = rbi_chatbot.ask_question(question, k)
    return result["answer"], result["sources"]

# 🌊 New streaming function
def ask_question_stream(question: str, k: int = 4) -> Iterator[Dict[str, Any]]:
    """Stream the response for a question"""
    return rbi_chatbot.ask_question_stream(question, k)

def test_streaming():
    """Test streaming functionality"""
    print("\n🌊 Testing Streaming Features")
    print("=" * 40)
    
    question = "What are the key RBI guidelines for banking?"
    print(f"Question: {question}")
    print("\nStreaming response:")
    
    for chunk in ask_question_stream(question):
        if chunk["type"] == "chunk":
            print(chunk["content"], end="", flush=True)
        elif chunk["type"] == "complete":
            print(f"\n\n✅ Streaming complete!")
            print(f"📊 Total sources: {len(chunk['sources'])}")
        elif chunk["type"] == "error":
            print(f"\n❌ Error: {chunk['content']}")

def test_context_features():
    """Test various context features"""
    print("\n🧪 Testing Context Features")
    print("=" * 40)
    
    # Test context summary
    summary = rbi_chatbot.get_context_summary()
    print("📊 Context Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Test recent context
    recent = rbi_chatbot.get_recent_context(2)
    print(f"\n📋 Recent Context (last 2 pairs): {len(recent)} pairs")
    
    # Test context window update
    rbi_chatbot.update_context_window(3)
    print(f"✅ Context window updated")

if __name__ == "__main__":
    # Run demos
    test_streaming()
    test_context_features()