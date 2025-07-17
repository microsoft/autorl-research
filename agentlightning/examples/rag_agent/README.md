# RAG agent example


1. Download the prepared FAISS index and chunks
2. Start retriever mcp server: `python server/wiki_retriever_mcp.py`
3. Start ray: `bash ../../scripts/restart_ray.sh`. To use Wandb, you need to set the WANDB_API_KEY environment variable before starting ray.
4. Run the agent: `python rag_agent.py`. For efficiency, it is recommended to run the agent with multiple workers, like `bash run_multiple_agents.sh`.
5. In another terminal, launch the training server: `bash train.sh`.
